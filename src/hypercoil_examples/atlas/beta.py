# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Distribution compatibility layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compatibility layer for representing beta and vMF distributions, because
numpyro is flaky and erratic when used with equinox (de)serialisation.
"""
from typing import Sequence
import equinox as eqx
from numpyro.distributions import Beta
from hypercoil.engine import Tensor
from hypercoil.init import VonMisesFisher


class BetaCompat(eqx.Module):
    concentration0: float
    concentration1: float
    
    def log_prob(self, *pparams, **params):
        return Beta(
            self.concentration0,
            self.concentration1,
        ).log_prob(*pparams, **params)


class VMFCompat(eqx.Module):
    mu: Tensor
    kappa: Tensor
    sample_max_iter: int = 5
    sample_return_valid: bool = False
    explicit_normalisation: bool = True
    parameterise: bool = True

    def sample(
        self,
        key: 'jax.random.PRNGKey',
        sample_shape: Sequence[int] = (),
    ) -> Tensor:
        return VonMisesFisher(
            mu=self.mu,
            kappa=self.kappa,
            sample_max_iter=self.sample_max_iter,
            sample_return_valid=self.sample_return_valid,
            explicit_normalisation=self.explicit_normalisation,
            parameterise=self.parameterise,
        ).sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return VonMisesFisher(
            mu=self.mu,
            kappa=self.kappa,
            sample_max_iter=self.sample_max_iter,
            sample_return_valid=self.sample_return_valid,
            explicit_normalisation=self.explicit_normalisation,
            parameterise=self.parameterise,
        ).log_prob(value=value)

