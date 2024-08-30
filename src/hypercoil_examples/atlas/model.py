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
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.special import logit
from numpyro.distributions import Distribution
from scipy.special import softmax

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.functional import (
    corr,
    residualise,
    spherical_geodesic,
    sym2vec,
)

from hypercoil_examples.atlas.beta import VMFCompat
from hypercoil_examples.atlas.ellgat import ELLGATCompat, ELLGATBlock
from hypercoil_examples.atlas.multiencoder import configure_multiencoder
from hypercoil_examples.atlas.positional import (
    configure_geometric_encoder, get_coors
)
from hypercoil_examples.atlas.promises import empty_promises
from hypercoil_examples.atlas.unet import get_meshes, IcoELLGATUNet

VMF_BASE_KAPPA = 10
ATTN_HEADS_ARCH = (4, 4, 4)
BLOCK_ARCH = {
    'ELLGAT': ELLGATCompat,
    'ELLGATBlock': ELLGATBlock,
}
ENCODER_IN_DIM_ARCH = {
    '64x64': 64,
    '3res': 14,
}
FORWARD_IN_DIM_ARCH = {
    'ELLGAT': (64, 256),
    'ELLGATBlock': (32, 64),
}
HIDDEN_DIM_ARCH = {
    'ELLGAT': (16, 64, 256),
    'ELLGATBlock': (8, 16, 64),
}
CONTRALATERAL_JITTER = 1e-4
NUM_ITER_VMFMM_INIT = 10
TEMPLATE_PATH = '/tmp/mean_init.npy'


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
        denom = T.std(-1, keepdims=True)
        denom = jnp.where(denom == 0, 1., denom)
        T = T / denom
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
    doublet_potential: callable
    mass_potential: Optional[callable] = None
    spatial_mle: Optional[callable] = None
    selectivity_mle: Optional[callable] = None

    def __init__(
        self,
        spatial_distribution: Distribution,
        selectivity_distribution: Distribution,
        doublet_potential: callable,
        mass_potential: Optional[callable] = None,
        spatial_mle: Optional[callable] = None,
        selectivity_mle: Optional[callable] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.spatial_distribution = spatial_distribution
        self.selectivity_distribution = selectivity_distribution
        self.doublet_potential = doublet_potential
        self.mass_potential = mass_potential
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
        result = self.doublet_potential(coassignment)
        return jnp.where(D < 0, 0., result)

    def expected_energy(
        self,
        Q: Tensor,
        U: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        D: Optional[Tensor] = None,
        *,
        point_nu: float = 1.,
        doublet_nu: float = 1.,
        mass_nu: float = 1.,
    ) -> Tensor:
        energy = jnp.asarray(0)
        # point energies
        if U is not None:
            energy += point_nu * (Q * U).sum(-1)
        elif Z is not None and S is not None:
            energy += point_nu * (Q * self.point_energy(Z=Z, S=S)).sum(-1)
        # doublet energies
        if D is not None:
            energy += doublet_nu * self.doublet_energy(
                D=D,
                Q=jnp.clip(Q, 1e-6, 1 - 1e-6),
            ).sum(-1)
        if self.mass_potential is not None:
            energy += mass_nu * self.mass_potential(Q.sum(-2))
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
            doublet_potential=self.doublet_potential,
            mass_potential=self.mass_potential,
            spatial_mle=self.spatial_mle,
            selectivity_mle=self.selectivity_mle,
        )

    def __call__(
        self,
        Z: Tensor,
        S: Tensor,
        D: Optional[Tensor] = None,
        temperature: float = 1.,
    ) -> Tensor:
        U = self.point_energy(Z=Z, S=S)
        Q = jax.nn.softmax(-U / temperature, axis=-1)
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
        temperature: float = 1.,
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
            compartment: jax.nn.softmax(
                -point[compartment] / temperature,
                axis=-1,
            )
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
        point_nu: float = 1.,
        doublet_nu: float = 1.,
        mass_nu: float = 1.,
        temperature: float = 1.,
        # The below arguments are here only for uniformity of the interface.
        # They do nothing.
        encoder_type: Literal['64x64', '3res'] = '64x64',
        injection_points: Sequence[
            Literal['input', 'readout', 'residual']
        ] = (),
        additional_data: Sequence[
            Literal['geometric', 'parametric']
        ] = ('geometric',),
        inference: Optional[bool] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        # The third argument is the "positional" / geometric encoding, which
        # only the approximator (U-Net) uses.
        regulariser, X, _, compartments, (M, new_M, new_update_weight) = (
            self._common_path(
                coor=coor,
                T=T,
                M=M,
                new_M=new_M,
                update_weight=update_weight,
                encoder=encoder,
                encoder_result=encoder_result,
                compartments=compartments,
                temperature=temperature,
            )
        )
        P = {
            compartment: jax.nn.softmax(
                -regulariser[compartment].point_energy(
                    Z=X[compartment],
                    S=coor[compartment],
                ).swapaxes(-1, -2) / temperature,
                axis=-2,
            )
            for compartment in compartments
        }
        energy = {
            compartment: regulariser[compartment].expected_energy(
                Q=P[compartment].swapaxes(-1, -2),
                S=coor[compartment],
                Z=X[compartment],
                D=self.approximator.meshes[compartment].icospheres[0],
                point_nu=point_nu,
                doublet_nu=doublet_nu,
                mass_nu=mass_nu,
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
        point_nu: float = 1.,
        doublet_nu: float = 1.,
        mass_nu: float = 1.,
        encoder_type: Literal['64x64', '3res'] = '64x64',
        injection_points: Sequence[
            Literal['input', 'readout', 'residual']
        ] = (),
        # DO NOT set both 'parametric' here and 'input' in injection_points.
        # Otherwise, the results of the parametric model (regulariser) will be
        # duplicated in the input to the approximator.
        additional_data: Sequence[
            Literal['geometric', 'parametric']
        ] = ('geometric',),
        temperature: float = 1.,
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
                temperature=temperature,
            )
        )

        if len(injection_points) > 0 or 'parametric' in additional_data:
            energy = {
                compartment: regulariser[compartment].point_energy(
                    Z=X[compartment],
                    S=coor[compartment],
                ).swapaxes(-2, -1)
                for compartment in compartments
            }
            P = {
                compartment: jax.nn.softmax(
                    -energy[compartment] / temperature,
                    axis=-2,
                )
                for compartment in compartments
            }
        if additional_data:
            agg_fn = lambda x, y: jnp.concatenate((x, *y), axis=-1)
        else:
            agg_fn = lambda x, y: x
        additional_data = {
            compartment: tuple(
                e for e in (
                    S[compartment] if 'geometric' in additional_data else None,
                    P[compartment] if 'parametric' in additional_data else None,
                )
                if e is not None
            )
            for compartment in compartments
        }

        masks = encoder.temporal.reduced_slices[::-1]
        inputs = {
            compartment: tuple(
                agg_fn(
                    jax.lax.dynamic_slice(
                        X[compartment],
                        (0,) * (X[compartment].ndim - 1) + (s[0],),
                        X[compartment].shape[:-1] + (s[1],),
                    ),
                    additional_data[compartment],
                ).swapaxes(-1, -2)
                for s in masks
            )
            for compartment in compartments
        }
        if encoder_type == '64x64':
            inputs = {k: (v[0], v[0], v[1]) for k, v in inputs.items()}
        if len(injection_points) > 0:
            if 'input' in injection_points:
                inputs = {
                    compartment: (
                        jnp.concatenate(
                            (inputs[compartment][0], P[compartment])
                        ),
                        *inputs[compartment][1:],
                    )
                    for compartment in compartments
                }
            inputs = {
                compartment: (
                    *inputs[compartment],
                    (
                        P[compartment]
                        if 'readout' in injection_points
                        else None
                    ),
                    (
                        -energy[compartment]
                        if 'residual' in injection_points
                        else None
                    ),
                )
                for compartment in compartments
            }
        P = {
            compartment: self.approximator(
                inputs[compartment],
                mesh=compartment,
                inference=inference,
                temperature=temperature,
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
                point_nu=point_nu,
                doublet_nu=doublet_nu,
                mass_nu=mass_nu,
            )
            for compartment in compartments
        }
        energy = {
            compartment: jnp.mean(energy[compartment])
            for compartment in compartments
        }
        # energy = jnp.stack(
        #     tuple(
        #         jnp.mean(energy[compartment])
        #         for compartment in compartments
        #     )
        # ).mean()
        return P, energy, (M, new_M, new_update_weight)


def vmf_mle_mu_only(
    Q: Tensor,
    data: Tensor,
    norm_f: Optional[callable] = None,
    src_distr: Optional[VMFCompat] = None,
) -> VMFCompat:
    mu = Q.T @ data
    if norm_f is None:
        mu = mu / jnp.linalg.norm(mu, axis=-1, keepdims=True)
    else:
        mu = norm_f(mu)
    if src_distr is not None:
        kappa = src_distr.kappa
    else:
        kappa = VMF_BASE_KAPPA
    return VMFCompat(mu=mu, kappa=kappa, parameterise=False)


def marginal_entropy(counts: Tensor) -> Tensor:
    P = counts / counts.sum(-1, keepdims=True)
    P = jnp.clip(P, 1e-6, 1 - 1e-6)
    return -jnp.sum(P * jnp.log(P), axis=-1)


def logit_normal_divergence(
    P: Tensor,
    Q: Tensor,
    scale: Tensor = 1.,
) -> Tensor:
    return jnp.exp(
        -(
            (
                logit(jnp.clip(P, 1e-6, 1 - 1e-6)) -
                logit(jnp.clip(Q, 1e-6, 1 - 1e-6))
            ) / scale
        ) ** 2
    )


def logit_normal_maxent_divergence(P: Tensor, scale: Tensor = 1.) -> Tensor:
    maxent = 1 / P.shape[-2]
    return logit_normal_divergence(P, maxent, scale=scale)


def _apply_over_compartments(
    fn: callable,
    compartments: Tuple[str],
    *data: Mapping[str, Tensor],
    **params,
) -> Mapping[str, Tensor]:
    return {
        compartment: fn(
            data[0][compartment],
            *[e[compartment] for e in data[1:]],
            **params,
        )
        for compartment in compartments
    }


def kappa_energy(
    parametric: SpatialSelectiveMRF,
    /,
    *,
    spatial_energy: float,
    selectivity_energy: float,
) -> Tensor:
    energy = 0
    for kappa, kappa_energy in (
        (
            parametric.spatial_distribution.kappa,
            spatial_energy,
        ),
        (
            parametric.selectivity_distribution.kappa,
            selectivity_energy,
        ),
    ):
        if kappa_energy is not None:
            prior, err_large, err_small = kappa_energy
            kappa_err = kappa - prior
            energy = energy + jax.lax.cond(
                kappa_err < 0,
                lambda err: -err_small * err,
                lambda err: err_large * err,
                kappa_err,
            ).mean()
    return energy


def reconstruction_error(
    P: Tensor,
    limits: Tuple[int, int],
    /,
    *,
    T: Tensor,
    recon_nu: float,
) -> Tensor:
    start, size = limits
    Tc = T[..., start:(start + size), :]
    parcel_ts = jnp.linalg.lstsq(
        P.swapaxes(-2, -1),
        Tc,
    )[0]
    return recon_nu * jnp.linalg.norm(
        P.swapaxes(-2, -1) @ parcel_ts - Tc
    )


def tether_energy(
    ipsilateral_model: SpatialSelectiveMRF,
    contralateral_model: SpatialSelectiveMRF,
    key: 'jax.random.PRNGKey',
    /,
    *,
    tether_nu: float,
) -> Tensor:
    contralateral = _to_jax_array(
        contralateral_model.spatial_distribution.mu
    )
    #TODO: We can get NaN if the two are aligned exactly. Here we add noise
    #      to avoid this edge case, but we should work out why this occurs.
    contralateral = contralateral + CONTRALATERAL_JITTER * jax.random.normal(
        key, contralateral.shape
    )
    contralateral = contralateral / jnp.linalg.norm(
        contralateral, axis=-1, keepdims=True
    )
    return tether_nu * spherical_geodesic(
        _to_jax_array(
            ipsilateral_model.spatial_distribution.mu
        )[..., None, :],
        contralateral.at[..., 0].set(-contralateral[..., 0])[..., None, :],
    ).sum()


def parcellate(
    P: Tensor,
    limits: Tuple[int, int],
    *,
    T: Tensor,
) -> Tensor:
    start, size = limits
    Tc = T[..., start:(start + size), :]
    return jnp.linalg.lstsq(
        P.swapaxes(-2, -1),
        Tc,
    )[0]


def init_encoder_model(
    coor_L: Tensor,
    encoder_type: Literal['64x64', '3res'] = '64x64',
):
    import numpy as np
    from hypercoil_examples.atlas.spatialinit import init_spatial_priors
    temporal_encoder = configure_multiencoder(
        use_7net=(encoder_type == '3res')
    )
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
    template = np.load(TEMPLATE_PATH)
    template = encoder.temporal.rescale(template)
    # template = template / jnp.linalg.norm(template, axis=-1, keepdims=True)
    template = {
        'cortex_L': template[:size_left],
        'cortex_R': template[size_left:],
    }
    return encoder, template


def refine_parcels(
    spatial_coor_parcels: Mapping[str, Tensor],
    spatial_coor: Mapping[str, Tensor],
    selectivity_coor_parcels: Mapping[str, Tensor],
    selectivity_coor: Mapping[str, Tensor],
    *,
    num_iter: int = NUM_ITER_VMFMM_INIT,
    spatial_kappa: float = VMF_BASE_KAPPA,
    selectivity_kappa: float = VMF_BASE_KAPPA,
) -> Tuple[Tensor, Tensor]:
    # There's absolutely no convergence guarantee here.
    # (Nor do we need it, really. We're only using this to initialise the
    # model. We just let it explore a bit and then take the best result.)
    # We should at least do this jointly, but I guess this alternating method
    # is probably better than nothing.
    compartments = spatial_coor.keys()
    spatial_coor_parcels_best = None
    selectivity_coor_parcels_best = None
    best_delta = float('inf')
    for i in range(num_iter):
        spatial_distribution = {
            compartment: VMFCompat(
                mu=spatial_coor_parcels[compartment],
                kappa=spatial_kappa,
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        selectivity_expect = {
            compartment: jax.nn.softmax(
                spatial_distribution[compartment].log_prob(
                    spatial_coor[compartment]
                ),
                axis=-1,
            )
            for compartment in compartments
        }
        selectivity_coor_parcels_new = {
            compartment: (
                selectivity_expect[compartment].T @
                selectivity_coor[compartment]
            )
            for compartment in compartments
        }
        selectivity_coor_parcels_new = {
            compartment: (
                selectivity_coor_parcels_new[compartment] / jnp.linalg.norm(
                    selectivity_coor_parcels_new[compartment],
                    axis=-1,
                    keepdims=True,
                )
            )
            for compartment in compartments
        }
        selectivity_delta = sum([
            jnp.linalg.norm(
                selectivity_coor_parcels_new[compartment] -
                selectivity_coor_parcels[compartment]
            ).item()
            for compartment in compartments
        ])
        selectivity_coor_parcels = selectivity_coor_parcels_new
        selectivity_distribution = {
            compartment: VMFCompat(
                mu=selectivity_coor_parcels[compartment],
                kappa=selectivity_kappa,
            )
            for compartment in ('cortex_L', 'cortex_R')
        }
        spatial_expect = {
            compartment: jax.nn.softmax(
                selectivity_distribution[compartment].log_prob(
                    selectivity_coor[compartment]
                ),
                axis=-1,
            )
            for compartment in compartments
        }
        spatial_coor_parcels_new = {
            compartment: (
                spatial_expect[compartment].T @ spatial_coor[compartment]
            )
            for compartment in compartments
        }
        spatial_coor_parcels_new_compl = {
            compartment: jnp.concatenate(
                (
                    -spatial_coor_parcels_new[other][..., :1],
                    spatial_coor_parcels_new[other][..., 1:],
                ),
                axis=-1,
            )
            for compartment, other in (
                ('cortex_L', 'cortex_R'),
                ('cortex_R', 'cortex_L'),
            )
        }
        spatial_coor_parcels_new = {
            compartment: (
                spatial_coor_parcels_new[compartment] +
                spatial_coor_parcels_new_compl[compartment] / 2
            )
            for compartment in compartments
        }
        spatial_coor_parcels_new = {
            compartment: (
                spatial_coor_parcels_new[compartment] / jnp.linalg.norm(
                    spatial_coor_parcels_new[compartment],
                    axis=-1,
                    keepdims=True,
                )
            )
            for compartment in compartments
        }
        spatial_delta = sum([
            jnp.linalg.norm(
                spatial_coor_parcels_new[compartment] -
                spatial_coor_parcels[compartment]
            ).item()
            for compartment in compartments
        ])
        spatial_coor_parcels = spatial_coor_parcels_new
        delta = selectivity_delta + spatial_delta
        if delta < best_delta:
            spatial_coor_parcels_best = spatial_coor_parcels
            selectivity_coor_parcels_best = selectivity_coor_parcels
        print(f'Iteration {i}: delta {delta}')
    return (
        spatial_coor_parcels_best,
        selectivity_coor_parcels_best,
    )


def init_full_model(
    T: Tensor,
    coor_L: Tensor,
    coor_R: Tensor,
    num_parcels: int = 100,
    encoder_type: Literal['64x64', '3res'] = '64x64',
    injection_points: Sequence[Literal['input', 'readout', 'residual']] = (),
    additional_data: Sequence[
        Literal['geometric', 'parametric']
    ] = ('geometric',),
    block_arch: Literal['ELLGAT', 'ELLGATBlock'] = 'ELLGATBlock',
    spatial_kappa: float = VMF_BASE_KAPPA,
    selectivity_kappa: float = VMF_BASE_KAPPA,
    fixed_kappa: bool = True,
    disaffiliation_penalty: float = 1.,
    dropout: float = 0.1,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tuple[eqx.Module, eqx.Module]:
    import numpy as np
    if key is None:
        key = jax.random.PRNGKey(0)
    key_r, key_a = jax.random.split(key)
    encoder, template = init_encoder_model(coor_L, encoder_type=encoder_type)
    result = encoder(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        M=template,
    )
    coor_parcels_L = coor_L[jax.random.choice(
        key_r,
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
    (spatial_coor_parcels, selectivity_coor_parcels) = refine_parcels(
        spatial_coor_parcels={
            'cortex_L': coor_parcels_L,
            'cortex_R': coor_parcels_R,
        },
        spatial_coor={
            'cortex_L': coor_L,
            'cortex_R': coor_R,
        },
        selectivity_coor_parcels=selectivity_parcels,
        selectivity_coor=template,
        spatial_kappa=spatial_kappa,
        selectivity_kappa=selectivity_kappa,
        num_iter=NUM_ITER_VMFMM_INIT,
    )
    if not fixed_kappa:
        selectivity_kappa = jnp.asarray(num_parcels * [selectivity_kappa], dtype=float)
        spatial_kappa = jnp.asarray(num_parcels * [spatial_kappa], dtype=float)
    selectivity_distribution = {
        compartment: VMFCompat(
            mu=selectivity_coor_parcels[compartment],
            kappa=selectivity_kappa,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    spatial_distribution = {
        compartment: VMFCompat(
            mu=spatial_coor_parcels[compartment],
            kappa=spatial_kappa,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    regulariser = {
        compartment: SpatialSelectiveMRF(
            spatial_distribution=spatial_distribution[compartment],
            selectivity_distribution=selectivity_distribution[compartment],
            doublet_potential=lambda x: logit_normal_divergence(
                x,
                1 / num_parcels,
                scale=5.,
            ) - x * disaffiliation_penalty,
            mass_potential=lambda x: -marginal_entropy(x),
            spatial_mle=vmf_mle_mu_only,
            selectivity_mle=vmf_mle_mu_only,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    readout_skip_dim = 0
    positional_dim = None
    base_in_dim = ENCODER_IN_DIM_ARCH[encoder_type]
    if 'input' in injection_points:
        base_in_dim += num_parcels
    if 'readout' in injection_points:
        readout_skip_dim = num_parcels
    if 'geometric' in additional_data:
        positional_dim = encoder.spatial.default_encoding_dim
        base_in_dim += positional_dim
    if 'parametric' in additional_data:
        base_in_dim += num_parcels
    mesh_L, mesh_R = get_meshes(
        model='full',
        positional_dim=positional_dim,
    )
    in_dim = (base_in_dim,) + FORWARD_IN_DIM_ARCH[block_arch]
    hidden_dim = HIDDEN_DIM_ARCH[block_arch]
    approximator = IcoELLGATUNet(
        meshes={
            'cortex_L': mesh_L,
            'cortex_R': mesh_R,
        },
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        hidden_readout_dim=num_parcels // 2,
        attn_heads=ATTN_HEADS_ARCH,
        block_arch=BLOCK_ARCH[block_arch],
        readout_dim=num_parcels,
        dropout=dropout,
        dropout_inference=False,
        readout_skip_dim=readout_skip_dim,
        key=key_a,
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
    compartments: Sequence[str],
    mode: Literal['full', 'regulariser'] = 'full',
    energy_nu: float = 1.,
    recon_nu: float = 1.,
    tether_nu: float = 1.,
    div_nu: float = 1e3,
    template_energy_nu: float = 1.,
    point_potentials_nu: float = 1.,
    doublet_potentials_nu: float = 1.,
    mass_potentials_nu: float = 1.,
    spatial_kappa_energy: Optional[Tuple[float, float, float]] = None,
    selectivity_kappa_energy: Optional[Tuple[float, float, float]] = None,
    linear_classifier_nu: float = 0.,
    linear_classifier_target: Optional[Tensor] = None,
    readout_name: Optional[str] = None,
    temperature: float = 1.,
    inference: bool = False,
    encoder_type: Literal['64x64', '3res'] = '64x64',
    injection_points: Sequence[Literal['input', 'readout', 'residual']] = (),
    additional_data: Sequence[
        Literal['geometric', 'parametric']
    ] = ('geometric',),
    key: 'jax.random.PRNGKey',
):
    meta = {compartment: {} for compartment in compartments}
    key_m, key_n = jax.random.split(key, 2)
    if mode == 'full':
        fwd = model
    else:
        fwd = model.regulariser_path
    result = fwd(
        coor=coor,
        encoder=encoder,
        encoder_result=encoder_result,
        compartments=compartments,
        inference=inference,
        point_nu=point_potentials_nu,
        doublet_nu=doublet_potentials_nu,
        mass_nu=mass_potentials_nu,
        encoder_type=encoder_type,
        temperature=temperature,
        injection_points=injection_points,
        additional_data=additional_data,
        key=key_m,
    )
    (_, (_, _, _, _, T)) = encoder_result
    P, energy, _ = result
    energy = _apply_over_compartments(
        lambda x: energy_nu * x, compartments, energy,
    )
    kappa_energies = _apply_over_compartments(
        kappa_energy,
        compartments,
        model.regulariser,
        spatial_energy=spatial_kappa_energy,
        selectivity_energy=selectivity_kappa_energy,
    )
    if energy_nu != 0:
        for compartment in compartments:
            meta[compartment]['energy'] = energy[compartment]
    if spatial_kappa_energy is not None:
        for compartment in compartments:
            meta[compartment]['spatial_kappa_energy'] = (
                kappa_energies[compartment]
            )
    if recon_nu != 0:
        recon_error = _apply_over_compartments(
            reconstruction_error,
            compartments,
            P,
            encoder.temporal.encoders[0].limits,
            T=T,
            recon_nu=recon_nu,
        )
        for compartment in compartments:
            meta[compartment]['recon_error'] = recon_error[compartment]
    else:
        recon_error = {k: 0 for k in compartments}
    if tether_nu != 0:
        key_n = jax.random.split(key_n, len(compartments))
        key_n = {k: key_n[i] for i, k in enumerate(compartments)}
        tether_loss = _apply_over_compartments(
            tether_energy,
            compartments,
            model.regulariser,
            {
                'cortex_L': model.regulariser['cortex_R'],
                'cortex_R': model.regulariser['cortex_L'],
            },
            key_n,
            tether_nu=tether_nu,
        )
        for compartment in compartments:
            meta[compartment]['tether_loss'] = tether_loss[compartment]
    else:
        tether_loss = {k: 0 for k in compartments}
    if div_nu != 0:
        div = _apply_over_compartments(
            lambda x: div_nu * logit_normal_maxent_divergence(x).mean(),
            compartments,
            P,
        )
        for compartment in compartments:
            meta[compartment]['div'] = div[compartment]
    else:
        div = {k: 0 for k in compartments}
    if template_energy_nu != 0:
        M = encoder_result[1][0]
        template_energy = {
            compartment: template_energy_nu * model.regulariser[compartment](
                M[compartment],
                coor[compartment],
                model.approximator.meshes[compartment].icospheres[0],
            ).mean()
            for compartment in compartments
        }
        for compartment in compartments:
            meta[compartment]['template_energy'] = (
                template_energy[compartment]
            )
    else:
        template_energy = {k: 0 for k in compartments}
    if linear_classifier_nu != 0:
        parcel_ts = _apply_over_compartments(
            parcellate,
            compartments,
            P,
            encoder.temporal.encoders[0].limits,
            T=T,
        )
        parcel_ts = jnp.concatenate(
            tuple(parcel_ts[compartment] for compartment in compartments),
            axis=-2,
        )
        parcel_adjvec = sym2vec(corr(parcel_ts))
        prediction = jax.nn.log_softmax(
            model.readouts[readout_name] @ parcel_adjvec
        )
        pred_error = -(linear_classifier_target * prediction).sum()
        for compartment in compartments:
            meta[compartment]['prediction_error'] = pred_error
    else:
        pred_error = {k: 0 for k in compartments}
    return sum(
        sum(meta[compartment].values())
        for compartment in compartments
    ), meta


def main(subject: str = '01', session: str = '01', num_parcels: int = 100):
    from hypercoil_examples.atlas.aligned_dccc import (
        get_msc_dataset, _get_data
    )
    coor_L, coor_R = get_coors()
    T = _get_data(
        get_msc_dataset(subject, session),
        normalise=False,
        gsr=False,
    )
    model, encoder, template = init_full_model(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        num_parcels=num_parcels,
    )
    print('Model initialised!')
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
        compartments=('cortex_L', 'cortex_R'),
        key=jax.random.PRNGKey(0),
    )
    assert 0


if __name__ == '__main__':
    main()
