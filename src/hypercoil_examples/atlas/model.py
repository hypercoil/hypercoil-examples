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
from scipy.special import softmax

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.functional import residualise, spherical_geodesic
from hypercoil.init import VonMisesFisher
from hypercoil.loss.functional import entropy

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

ELLGAT_DROPOUT = 0.6


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
        return (
            (X, X_aligned, S),
            (
                M,
                new_M,
                new_update_weight,
                self.temporal.rescale,
                T,
            ),
        )


class SpatialSelectiveMRF(eqx.Module):
    spatial_distribution: Distribution
    selectivity_distribution: Distribution
    doublet_distribution: Distribution
    mass_distribution: Optional[Distribution] = None
    spatial_mle: Optional[callable] = None
    selectivity_mle: Optional[callable] = None

    def __init__(
        self,
        spatial_distribution: Distribution,
        selectivity_distribution: Distribution,
        doublet_distribution: Distribution,
        mass_distribution: Optional[Distribution] = None,
        spatial_mle: Optional[callable] = None,
        selectivity_mle: Optional[callable] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.spatial_distribution = spatial_distribution
        self.selectivity_distribution = selectivity_distribution
        self.doublet_distribution = doublet_distribution
        self.mass_distribution = mass_distribution
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
        energy = jnp.asarray(0)
        # point energies
        if U is not None:
            energy += (Q * U).sum(-1)
        elif Z is not None and S is not None:
            energy += (Q * self.point_energy(Z=Z, S=S)).sum(-1)
        # doublet energies
        if D is not None:
            energy += self.doublet_energy(D=D, Q=Q).sum(-1)
        if self.mass_distribution is not None:
            energy += -self.mass_distribution.log_prob(Q.sum(-2))
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
            mass_distribution=self.mass_distribution,
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

    def _common_path(
        self,
        coor: Mapping[str, Tensor],
        T: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
        new_M: Optional[Tensor] = None,
        update_weight: int = 0,
        *,
        encoder: Optional[eqx.Module] = None,
        encoder_result: Optional[Tuple[Tuple, Tuple]] = None,
        compartments: Union[str, Tuple[str]] = ('cortex_L', 'cortex_R'),
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(compartments, str):
            compartments = (compartments,)
        if encoder_result is None:
            encoder_result = jax.lax.stop_gradient(
                encoder(
                    T=T,
                    coor_L=coor['cortex_L'],
                    coor_R=coor['cortex_R'],
                    M=M,
                    new_M=new_M,
                    update_weight=update_weight,
                )
            )
        if encoder_result is None:
            raise ValueError(
                'Either encoder or encoder_result must be provided.'
            )
        (
            (X, X_aligned, S),
            (M, new_M, new_update_weight, rescale, _),
        ) = encoder_result
        point = {
            compartment: self.regulariser[compartment].point_energy(
                Z=X_aligned[compartment],
                S=coor[compartment],
            )
            for compartment in compartments
        }
        Q = {
            compartment: jax.nn.softmax(-point[compartment], axis=-1)
            for compartment in compartments
        }
        regulariser = {
            compartment: self.regulariser[compartment].point_mle(
                Q=Q[compartment],
                selectivity_data=X[compartment],
                spatial_data=coor[compartment],
                selectivity_norm_f=rescale,
            )
            for compartment in compartments
        }
        return regulariser, X, S, compartments, (M, new_M, new_update_weight)

    def regulariser_path(
        self,
        coor: Mapping[str, Tensor],
        T: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
        new_M: Optional[Tensor] = None,
        update_weight: int = 0,
        *,
        encoder: Optional[eqx.Module] = None,
        encoder_result: Optional[Tuple[Tuple, Tuple]] = None,
        compartments: Union[str, Tuple[str]] = ('cortex_L', 'cortex_R'),
        inference: Optional[bool] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        regulariser, X, S, compartments, (M, new_M, new_update_weight) = (
            self._common_path(
                coor=coor,
                T=T,
                M=M,
                new_M=new_M,
                update_weight=update_weight,
                encoder=encoder,
                encoder_result=encoder_result,
                compartments=compartments,
            )
        )
        P = {
            compartment: jax.nn.softmax(
                -regulariser[compartment].point_energy(
                    Z=X[compartment],
                    S=coor[compartment],
                ).swapaxes(-1, -2),
            axis=-2)
            for compartment in compartments
        }
        energy = {
            compartment: regulariser[compartment].expected_energy(
                Q=P[compartment].swapaxes(-1, -2),
                S=coor[compartment],
                Z=X[compartment],
                D=self.approximator.meshes[compartment].icospheres[0],
            )
            for compartment in compartments
        }
        energy = jnp.stack(
            tuple(
                jnp.mean(energy[compartment])
                for compartment in compartments
            )
        ).mean()
        return P, energy, (M, new_M, new_update_weight)

    def __call__(
        self,
        coor: Mapping[str, Tensor],
        T: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
        new_M: Optional[Tensor] = None,
        update_weight: int = 0,
        *,
        encoder: Optional[eqx.Module] = None,
        encoder_result: Optional[Tuple[Tuple, Tuple]] = None,
        compartments: Union[str, Tuple[str]] = ('cortex_L', 'cortex_R'),
        inference: Optional[bool] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        regulariser, X, S, compartments, (M, new_M, new_update_weight) = (
            self._common_path(
                coor=coor,
                T=T,
                M=M,
                new_M=new_M,
                update_weight=update_weight,
                encoder=encoder,
                encoder_result=encoder_result,
                compartments=compartments,
            )
        )
        masks = encoder.temporal.reduced_slices[::-1]
        inputs = {
            compartment: tuple(
                jnp.concatenate((
                    jax.lax.dynamic_slice(
                        X[compartment],
                        (0,) * (X[compartment].ndim - 1) + (s[0],),
                        X[compartment].shape[:-1] + (s[1],),
                    ),
                    S[compartment]
                ), axis=-1).swapaxes(-1, -2)
                for s in masks
            )
            for compartment in compartments
        }
        P = {
            compartment: self.approximator(
                inputs[compartment],
                mesh=compartment,
                inference=inference,
                key=key,
            )
            for compartment in compartments
        }
        energy = {
            compartment: regulariser[compartment].expected_energy(
                Q=P[compartment].swapaxes(-1, -2),
                S=coor[compartment],
                Z=X[compartment],
                D=self.approximator.meshes[compartment].icospheres[0],
            )
            for compartment in compartments
        }
        energy = jnp.stack(
            tuple(
                jnp.mean(energy[compartment])
                for compartment in compartments
            )
        ).mean()
        return P, energy, (M, new_M, new_update_weight)


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
    return VonMisesFisher(mu=mu, kappa=kappa, parameterise=False)


def init_full_model(
    T: Tensor,
    coor_L: Tensor,
    coor_R: Tensor,
    num_parcels: int = 100,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tuple[eqx.Module, eqx.Module]:
    import numpy as np
    from hypercoil_examples.atlas.spatialinit import init_spatial_priors
    key = key or jax.random.PRNGKey(0)
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
    coor_parcels_L = coor_L[jax.random.choice(
        key,
        coor_L.shape[0],
        shape=(num_parcels,),
        replace=False,
    )]
    coor_parcels_R = np.concatenate(
        (-coor_parcels_L[..., :1], coor_parcels_L[..., 1:]),
        axis=-1,
    )
    selectivity_parcels = {
        compartment: softmax(
            np.exp({
                'cortex_L': coor_L @ coor_parcels_L.swapaxes(-2, -1),
                'cortex_R': coor_R @ coor_parcels_R.swapaxes(-2, -1),
            }[compartment]).T,
            axis=-1,
        ) @ result[0][0][compartment]
        for compartment in ('cortex_L', 'cortex_R')
    }
    selectivity_parcels = {
        compartment: selectivity_parcels[compartment] / jnp.linalg.norm(
            selectivity_parcels[compartment],
            axis=-1,
            keepdims=True,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    selectivity_distribution = {
        compartment: VonMisesFisher(
            mu=selectivity_parcels[compartment],
            kappa=10.,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    spatial_distribution = {
        compartment: VonMisesFisher(
            mu={
                'cortex_L': coor_parcels_L,
                'cortex_R': coor_parcels_R,
            }[compartment],
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
            mass_distribution=numpyro.distributions.DirichletMultinomial(
                jnp.ones(num_parcels).astype(int) * 10,
                total_count=template[compartment].shape[0],
            ),
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
        #in_dim=(14 + encoder.spatial.default_encoding_dim, 64, 512),
        in_dim=(14 + encoder.spatial.default_encoding_dim, 32, 64),
        #hidden_dim=(16, 64, 256),
        hidden_dim=(8, 16, 64),
        hidden_readout_dim=num_parcels // 2,
        #attn_heads=(4, 4, 4),
        attn_heads=(4, 4, 4),
        readout_dim=num_parcels,
        norm=UnitSphereNorm(),
        dropout=ELLGAT_DROPOUT,
        dropout_inference=False,
        key=jax.random.PRNGKey(0),
    )
    model = ForwardParcellationModel(
        regulariser=regulariser,
        approximator=approximator,
    )
    return model, encoder, template


def forward(
    model: ForwardParcellationModel,
    *,
    coor: Mapping[str, Tensor],
    encoder: StaticEncoder,
    encoder_result: Tuple,
    compartment: str,
    mode: Literal['full', 'regulariser'] = 'full',
    energy_nu: float = 1.,
    recon_nu: float = 1.,
    tether_nu: float = 1.,
    nkl_nu: float = 1e2,
    key: 'jax.random.PRNGKey',
):
    meta = {}
    key_m, key_n = jax.random.split(key, 2)
    if mode == 'full':
        fwd = model
    else:
        fwd = model.regulariser_path
    result = fwd(
        coor=coor,
        encoder=encoder,
        encoder_result=encoder_result,
        compartments=(compartment,),
        key=key_m,
    )
    (_, (_, _, _, _, T)) = encoder_result
    P, energy, _ = result
    energy = energy_nu * energy
    if energy_nu != 0:
        meta = {**meta, 'energy': energy}
    if recon_nu != 0:
        start, size = encoder.temporal.encoders[0].limits[compartment]
        T = T[..., start:(start + size), :]
        parcel_ts = jnp.linalg.lstsq(
            P[compartment].swapaxes(-2, -1),
            T,
        )[0]
        recon_error = recon_nu * jnp.linalg.norm(
            P[compartment].swapaxes(-2, -1) @ parcel_ts - T
        )
        meta = {**meta, 'recon_error': recon_error}
    else:
        recon_error = 0
    if tether_nu != 0:
        other_hemi = 'cortex_R' if compartment == 'cortex_L' else 'cortex_L'
        other_coor = _to_jax_array(
            model.regulariser[other_hemi].spatial_distribution.mu
        )
        #TODO: We can get NaN if the two are aligned exactly. Here we add noise
        #      to avoid this edge case, but we should work out why this occurs.
        other_coor = other_coor + 1e-4 * jax.random.normal(
            key_n, other_coor.shape
        )
        other_coor = other_coor / jnp.linalg.norm(
            other_coor, axis=-1, keepdims=True
        )
        tether = tether_nu * spherical_geodesic(
            _to_jax_array(
                model.regulariser[compartment].spatial_distribution.mu
            )[..., None, :],
            other_coor.at[..., 0].set(-other_coor[..., 0])[..., None, :],
        ).sum()
        meta = {**meta, 'hemisphere_tether': tether}
    else:
        tether = 0
    if nkl_nu != 0:
        nkl = nkl_nu * (
            jnp.log(P[compartment]) / P[compartment].shape[0]
        ).sum(0).mean()
        meta = {**meta, 'nkl': nkl}
    else:
        nkl = 0
    return energy + recon_error + tether + nkl, meta


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
    encoder_result = encoder(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        M=template,
    )
    result = eqx.filter_value_and_grad(
        eqx.filter_jit(forward),
        has_aux=True,
    )(
    #result = eqx.filter_value_and_grad(forward, has_aux=True)(
    #result = forward(
        model,
        coor={
            'cortex_L': coor_L,
            'cortex_R': coor_R,
        },
        encoder_result=encoder_result,
        encoder=encoder,
        compartment='cortex_L',
        key=jax.random.PRNGKey(0),
    )
    assert 0


if __name__ == '__main__':
    main()
