# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Beta compatibility layer
~~~~~~~~~~~~~~~~~~~~~~~~
Compatibility layer for representing beta distributions
"""
import equinox as eqx
from numpyro.distributions import Beta


class BetaCompat(eqx.Module):
    concentration0: float
    concentration1: float
    
    def log_prob(self, *pparams, **params):
        return Beta(
            self.concentration0,
            self.concentration1,
        ).log_prob(*pparams, **params)

