# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Full model
~~~~~~~~~~
The complete parcellation model, including the indirect (approximator) network
(ELLGAT U-Net), the direct (regulariser) Markov random field (MRF) model, the
energy function that combines the two, and the selectivity-space encoder that
constructs the inputs to both the U-Net and MRF models.
"""
from typing import Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import numpyro
from numpyro.distributions import Distribution

from hypercoil.engine import Tensor
from hypercoil.functional import residualise
from hypercoil.init import VonMisesFisher

from hypercoil_examples.atlas.aligned_dccc import param_bimodal_beta
from hypercoil_examples.atlas.ellgat import UnitSphereNorm
from hypercoil_examples.atlas.encoders import (
    create_icosphere_encoder,
    create_consensus_encoder,
    create_7net_encoder,
)
from hypercoil_examples.atlas.multiencoder import configure_multiencoder
from hypercoil_examples.atlas.positional import (
    configure_geometric_encoder, get_coors
)
from hypercoil_examples.atlas.promises import empty_promises
from hypercoil_examples.atlas.selectransform import (
    incomplete_mahalanobis_transform,
    logistic_mixture_threshold,
)
from hypercoil_examples.atlas.unet import get_meshes, IcoELLGATUNet


class EmptyPromises(eqx.Module):
    spatial_prior_loc: Union[Tensor, Mapping[str, Tensor]]
    spatial_prior_data: Union[Tensor, Mapping[str, Tensor]]
    spatial_prior_weight: float

    def __call__(
        self,
        X: Tensor,
        M: Tensor,
        new_M: Optional[Tensor] = None,
        update_weight: int = 0,
        geom: Optional[str] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, int]]:
        if isinstance(self.spatial_prior_loc, Mapping):
            spatial_prior_loc = self.spatial_prior_loc[geom]
        else:
            spatial_prior_loc = self.spatial_prior_loc
        if isinstance(self.spatial_prior_data, Mapping):
            spatial_prior_data = self.spatial_prior_data[geom]
        else:
            spatial_prior_data = self.spatial_prior_data
        X, (new_M, new_update_weight) = empty_promises(
            X=X,
            M=M,
            new_M=new_M,
            update_weight=update_weight,
            spatial_prior_loc=spatial_prior_loc,
            spatial_prior_data=spatial_prior_data,
            spatial_prior_weight=self.spatial_prior_weight,
            return_loading=False,
        )
        return X, (new_M, new_update_weight)


class StaticEncoder(eqx.Module):
    """
    Convenience container for model parameters that should not be learnable.
    These are used in practice for encoding the input data into a selectivity
    space.
    """
    spatial: eqx.Module
    temporal: eqx.Module
    alignment: eqx.Module

    def __call__(
        self,
        T: Tensor,
        coor_L: Tensor,
        coor_R: Tensor,
        M: Mapping[str, Tensor],
        new_M: Optional[Mapping[str, Tensor]] = None,
        update_weight: int = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        # 1. Global signal regression
        T = T - T.mean(-1, keepdims=True)
        T = T / T.std(-1, keepdims=True)
        T = jnp.where(jnp.isnan(T), 0, T)
        gs = T.mean(0, keepdims=True)
        T = residualise(T, gs)
        # 2. Temporal encode
        X = self.temporal(T)
        # 3. Spatial encode
        S = {}
        S['cortex_L'] = self.spatial(coor_L, geom='cortex_L')
        S['cortex_R'] = self.spatial(coor_R, geom='cortex_R')
        # 4. Alignment
        new_M = new_M or {}
        slices = self.temporal.encoders[0].limits
        X = X.swapaxes(-1, -2)
        X = {
            geom: jax.lax.dynamic_slice(
                X,
                (0,) * (X.ndim - 1) + (slices[geom][0],),
                (*X.shape[:-1], slices[geom][1]),
            ).swapaxes(-1, -2)
            for geom in ('cortex_L', 'cortex_R')
        }
        X_aligned = {}
        for geom in ('cortex_L', 'cortex_R'):
            X_aligned_geom, (new_M_geom, new_update_weight) = self.alignment(
                X=X[geom],
                M=M[geom],
                new_M=new_M.get(geom, None),
                update_weight=update_weight,
                geom=geom,
            )
            X_aligned[geom] = X_aligned_geom
            new_M[geom] = new_M_geom
        return (X, X_aligned, S), (M, new_M, new_update_weight)


class SpatialSelectiveMRF(eqx.Module):
    spatial_distribution: Distribution
    selectivity_distribution: Distribution
    doublet_distribution: Distribution
    spatial_mle: Optional[callable] = None
    selectivity_mle: Optional[callable] = None

    def __init__(
        self,
        spatial_distribution: Distribution,
        selectivity_distribution: Distribution,
        doublet_distribution: Distribution,
        spatial_mle: Optional[callable] = None,
        selectivity_mle: Optional[callable] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.spatial_distribution = spatial_distribution
        self.selectivity_distribution = selectivity_distribution
        self.doublet_distribution = doublet_distribution
        self.spatial_mle = spatial_mle
        self.selectivity_mle = selectivity_mle

    def point_energy(
        self,
        Z: Tensor,
        S: Tensor,
    ) -> Tensor:
        return -(
            self.selectivity_distribution.log_prob(Z) +
            self.spatial_distribution.log_prob(S)
        )

    def doublet_energy(
        self,
        D: Tensor,
        Q: Tensor,
    ) -> Tensor:
        """
        Compute energy of a distribution Q for doublets in D.
        """
        coassignment = jnp.einsum(
            '...snd,...sd->...sn',
            # jnp.where(
            #     (D < 0)[..., None],
            #     0.,
            #     Q[..., D, :],
            # ),
            Q[..., D, :],
            Q,
        )
        result = -self.doublet_distribution.log_prob(coassignment)
        return jnp.where(D < 0, 0., result)

    def expected_energy(
        self,
        Q: Tensor,
        U: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        D: Optional[Tensor] = None,
    ) -> Tensor:
        energy = 0
        # point energies
        if U is not None:
            energy += (Q * U).sum(-1)
        elif Z is not None and S is not None:
            energy += (Q * self.point_energy(Z=Z, S=S)).sum(-1)
        # doublet energies
        if D is not None:
            energy += self.doublet_energy(D=D, Q=Q).sum(-1)
        return energy

    def point_mle(
        self,
        Q: Tensor,
        selectivity_data: Tensor,
        spatial_data: Tensor,
        selectivity_norm_f: Optional[callable] = None,
        spatial_norm_f: Optional[callable] = None,
    ) -> Tensor:
        return type(self)(
            spatial_distribution=self.spatial_mle(
                Q,
                spatial_data,
                norm_f=spatial_norm_f,
                src_distr=self.spatial_distribution,
            ),
            selectivity_distribution=self.selectivity_mle(
                Q,
                selectivity_data,
                norm_f=selectivity_norm_f,
                src_distr=self.selectivity_distribution,
            ),
            doublet_distribution=self.doublet_distribution,
            spatial_mle=self.spatial_mle,
            selectivity_mle=self.selectivity_mle,
        )

    def __call__(
        self,
        Z: Tensor,
        S: Tensor,
        D: Optional[Tensor] = None,
    ) -> Tensor:
        U = self.point_energy(Z=Z, S=S)
        Q = jax.nn.softmax(-U, axis=-1)
        return self.expected_energy(Q=Q, U=U, D=D)


class ForwardParcellationModel(eqx.Module):
    regulariser: Mapping[str, eqx.Module]
    approximator: eqx.Module

    def __call__(
        self,
        T: Tensor,
        coor: Mapping[str, Tensor],
        M: Tensor,
        new_M: Optional[Tensor] = None,
        update_weight: int = 0,
        *,
        encoder: eqx.Module,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        (X, X_aligned, S), (M, new_M, new_update_weight) = jax.lax.stop_gradient(
            encoder(
                T=T,
                coor_L=coor['cortex_L'],
                coor_R=coor['cortex_R'],
                M=M,
                new_M=new_M,
                update_weight=update_weight,
            )
        )
        point = {
            compartment: self.regulariser[compartment].point_energy(
                Z=X_aligned[compartment],
                S=coor[compartment],
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        Q = {
            compartment: jax.nn.softmax(-point[compartment], axis=-1)
            for compartment in ('cortex_L', 'cortex_R')
        }
        regulariser = {
            compartment: self.regulariser[compartment].point_mle(
                Q=Q[compartment],
                selectivity_data=X[compartment],
                spatial_data=coor[compartment],
                selectivity_norm_f=encoder.temporal.rescale,
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        masks = encoder.temporal.reduced_mask.T[::-1]
        inputs = {
            compartment: tuple(
                jnp.concatenate((
                    X[compartment][..., mask],
                    S[compartment]
                ), axis=-1).swapaxes(-1, -2)
                for mask in masks
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        P = {
            compartment: self.approximator(
                inputs[compartment],
                mesh=compartment,
                key=key,
            ).swapaxes(-1, -2)
            for compartment in ('cortex_L', 'cortex_R')
        }
        energy = {
            compartment: regulariser[compartment].expected_energy(
                Q=P[compartment],
                S=coor[compartment],
                Z=X[compartment],
                D=self.approximator.meshes[compartment].icospheres[0],
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        energy = jnp.stack(
            tuple(
                jnp.mean(energy[compartment])
                for compartment in ('cortex_L', 'cortex_R')
            )
        ).mean()
        return P, energy


def vmf_mle_mu_only(
    Q: Tensor,
    data: Tensor,
    norm_f: Optional[callable] = None,
    src_distr: Optional[VonMisesFisher] = None,
) -> VonMisesFisher:
    mu = Q.T @ data
    if norm_f is None:
        mu = mu / jnp.linalg.norm(mu, axis=-1, keepdims=True)
    else:
        mu = norm_f(mu)
    if src_distr is not None:
        kappa = src_distr.kappa
    else:
        kappa = 10.
    return VonMisesFisher(mu=mu, kappa=kappa)


def init_full_model(
    T: Tensor,
    coor_L: Tensor,
    coor_R: Tensor,
    num_parcels: int = 100,
) -> Tuple[eqx.Module, eqx.Module]:
    import numpy as np
    from hypercoil_examples.atlas.spatialinit import init_spatial_priors
    temporal_encoder = configure_multiencoder()
    spatial_encoder = configure_geometric_encoder()
    spatial_loc_left, spatial_loc_right, spatial_data = init_spatial_priors()
    alignment = EmptyPromises(
        spatial_prior_loc={
            'cortex_L': spatial_loc_left,
            'cortex_R': spatial_loc_right,
        },
        spatial_prior_data=spatial_data,
        spatial_prior_weight=0.1,
    )
    encoder = StaticEncoder(
        spatial=spatial_encoder,
        temporal=temporal_encoder,
        alignment=alignment,
    )
    size_left = coor_L.shape[0]
    template = np.load('/tmp/mean_init.npy')
    template = encoder.temporal.rescale(template)
    # template = template / jnp.linalg.norm(template, axis=-1, keepdims=True)
    template = {
        'cortex_L': template[:size_left],
        'cortex_R': template[size_left:],
    }
    result = encoder(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        M=template,
    )
    selectivity_distribution = {
        compartment: VonMisesFisher(
            mu=result[0][0][compartment][:num_parcels],
            kappa=10.,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    spatial_distribution = {
        compartment: VonMisesFisher(
            mu={
                'cortex_L': coor_L,
                'cortex_R': coor_R,
            }[compartment][:num_parcels],
            kappa=10.,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    doublet_distribution = param_bimodal_beta(jnp.sqrt(num_parcels))
    regulariser = {
        compartment: SpatialSelectiveMRF(
            spatial_distribution=spatial_distribution[compartment],
            selectivity_distribution=selectivity_distribution[compartment],
            doublet_distribution=doublet_distribution,
            spatial_mle=vmf_mle_mu_only,
            selectivity_mle=vmf_mle_mu_only,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    mesh_L, mesh_R = get_meshes(
        model='full',
        positional_dim=encoder.spatial.default_encoding_dim,
    )
    approximator = IcoELLGATUNet(
        meshes={
            'cortex_L': mesh_L,
            'cortex_R': mesh_R,
        },
        in_dim=(14 + encoder.spatial.default_encoding_dim, 64, 512),
        hidden_dim=(16, 64, 256),
        hidden_readout_dim=num_parcels // 2,
        attn_heads=(4, 4, 4),
        readout_dim=num_parcels,
        norm=UnitSphereNorm(),
        key=jax.random.PRNGKey(0),
    )
    model = ForwardParcellationModel(
        regulariser=regulariser,
        approximator=approximator,
    )
    return model, encoder, template


def main(subject: str = '01', session: str = '01', num_parcels: int = 100):
    from hypercoil_examples.atlas.aligned_dccc import (
        get_msc_dataset, _get_data
    )
    coor_L, coor_R = get_coors()
    T = _get_data(get_msc_dataset(subject, session))
    model, encoder, template = init_full_model(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        num_parcels=num_parcels,
    )
    result = model(
        T=T,
        coor={
            'cortex_L': coor_L,
            'cortex_R': coor_R,
        },
        M=template,
        encoder=encoder,
        key=jax.random.PRNGKey(0),
    )
    assert 0


if __name__ == '__main__':
    main()
