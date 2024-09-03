# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
KRLS-T
~~~~~~
The kernel recursive least squares-tracker (KRLS-T) algorithm can be used for
online learning of a kernelised regression model in settings where the data
are non-stationary, for instance when the data are outputs from a neural
network that is being trained online. The KRLS-T algorithm is based on the
paper:

  Kernel recursive least-squares tracker for time-varying regression
  Steven Van Vaerenbergh, Miguel Lázaro-Gredilla, Ignacio Santamaria
  10.1109/TNNLS.2012.2200500

The idea for using this method comes from this overview of online learning
kernel methods from some of the same authors:
https://gtas.unican.es/files/pub/ch21_online_regression_with_kernels.pdf

And the code is based mostly on the numpy- and sklearn-based implementation
by Lucas Krauß (lckr on GitHub) and released under the MIT license:
https://github.com/lckr/PyKRLST

One difference between the original implementation and this one is that the
JAX JIT compiler must be re-invoked whenever the size of any of the arrays
changes. Accordingly, because we initialise arrays at their full final size,
this implementation is not as space-efficient as the original, but it should
be faster (at least when the dictionary fills up) and GPU-compatible.
"""
from typing import Literal, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.engine import Tensor


class Kernel(Protocol):
    def __call__(
        self,
        x: Tensor,
        y: Tensor | None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        ...


class RBFKernel(eqx.Module):
    length_scale: float

    def __call__(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if y is None:
            y = x
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        return jnp.exp(
            -(
                jnp.einsum('ij,ij->i', x, x)[:, None] +
                jnp.einsum('ij,ij->i', y, y) -
                2 * x @ y.T
            ) /
            (2 * self.length_scale ** 2)
        )


class CorrelationKernel(eqx.Module):
    def __call__(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if key is not None:
            key_x, key_y = jax.random.split(key)
            x = x + jax.random.normal(key_x, x.shape) * 1e-10
        x = jnp.atleast_2d(x)
        if y is None:
            return jnp.corrcoef(x)
        elif key is not None:
            y = y + jax.random.normal(key_y, y.shape) * 1e-10
        y = jnp.atleast_2d(y)
        return jnp.corrcoef(x, y)[:x.shape[-2], -y.shape[-2]:]


class KRLST(eqx.Module):
    kernel: Kernel
    forgetting_factor: float
    regularisation: float
    dictionary_size: int
    dictionary: Tensor | None = None
    time_index: Tensor | None = None
    forget_mode: Literal['B2P', 'UI'] = 'B2P'
    nlin: Optional[callable] = None
    _mu: Tensor | None = None
    _sigma: Tensor | None = None
    _Q: Tensor | None = None
    _noise_power_numerator: Tensor = jnp.asarray(0.)
    _noise_power_denominator: Tensor = jnp.asarray(1.)
    _jitter: float = 1e-10

    def __post_init__(self):
        assert self.forgetting_factor >= 0 and self.forgetting_factor <= 1, (
            'Forgetting factor must be between 0 and 1.'
        )
        assert self.regularisation >= 0, (
            'Regularisation must be non-negative.'
        )
        assert self.dictionary_size > 0, (
            'Dictionary size must be positive.'
        )
        assert self.forget_mode in ('B2P', 'UI'), (
            'Forget mode must be either "B2P" or "UI".'
        )

    def _initialise(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor = 0.,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> 'KRLST':
        y_dim = y.shape[-1] if y.ndim > 0 else 1
        y = y[..., None, None]
        K = self.kernel(x, key=key) + self._jitter
        Q = 1 / K # inverse of the kernel matrix K
        mu = (y * K) / (K + self.regularisation)
        sigma = K - ((K ** 2) / (K + self.regularisation))
        Q = jnp.zeros(
            (self.dictionary_size, self.dictionary_size)
        ).at[..., 0, 0].set(Q.squeeze())
        mu = jnp.zeros(
            (y_dim, self.dictionary_size, 1)
        ).at[..., 0, 0].set(mu.squeeze())
        sigma = jnp.zeros(
            (self.dictionary_size, self.dictionary_size)
        ).at[..., 0, 0].set(sigma.squeeze())

        time_index = jnp.full((self.dictionary_size), -1.).at[0].set(t)
        dictionary = jnp.zeros((self.dictionary_size, x.shape[-1]))
        dictionary = dictionary.at[0].set(x)
        _noise_power_numerator = jnp.asarray(
            y ** 2 / (K + self.regularisation)
        )
        _noise_power_denominator = jnp.asarray(1.)
        return KRLST(
            kernel=self.kernel,
            forgetting_factor=self.forgetting_factor,
            regularisation=self.regularisation,
            dictionary_size=self.dictionary_size,
            _mu=mu,
            _sigma=sigma,
            _Q=Q,
            dictionary=dictionary,
            time_index=time_index,
            forget_mode=self.forget_mode,
            _noise_power_numerator=_noise_power_numerator,
            _noise_power_denominator=_noise_power_denominator,
            _jitter=self._jitter,
            nlin=self.nlin,
        )

    @property
    def _dictionary_alloc(self) -> Tensor:
        return self.time_index >= 0

    @property
    def _x_dictionary_alloc(self) -> Tensor:
        return jnp.concatenate((self._dictionary_alloc, jnp.atleast_1d(True)))

    @property
    def _dictionary_alloc_outer(self) -> Tensor:
        return (
            self._dictionary_alloc[:, None] * self._dictionary_alloc[None, :]
        )

    @property
    def mu(self) -> Tensor:
        if self._dictionary_alloc.sum() < self.dictionary_size:
            return self._mu[self._dictionary_alloc, :]
        else:
            return self._mu

    @property
    def sigma(self) -> Tensor:
        if self._dictionary_alloc.sum() < self.dictionary_size:
            return self._sigma[self._dictionary_alloc][
                :, self._dictionary_alloc,
            ]
        else:
            return self._sigma

    @property
    def Q(self) -> Tensor:
        if self._dictionary_alloc.sum() < self.dictionary_size:
            return self._Q[self._dictionary_alloc][
                :, self._dictionary_alloc,
            ]
        else:
            return self._Q

    @property
    def noise_power_maximum_likelihood(self) -> Tensor:
        return self._noise_power_numerator / self._noise_power_denominator

    def _forget_B2P(
        self,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        K = self.kernel(
            self.dictionary,
            key=key,
        ) * self._dictionary_alloc_outer
        sigma = (
            self.forgetting_factor * self._sigma +
            (1 - self.forgetting_factor) * K
        )
        mu = jnp.sqrt(self.forgetting_factor) * self._mu
        return sigma, mu

    def _forget_UI(self) -> Tuple[Tensor, Tensor]:
        sigma = self._sigma / self.forgetting_factor
        return sigma, self._mu

    def observe(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> 'KRLST':
        if self.dictionary is None:
            return self._initialise(x, y, t, key=key)
        y = y[..., None, None]
        if key is not None:
            key_f, key_dx, key_xx = jax.random.split(key, 3)
        else:
            key_f, key_dx, key_xx = None, None, None
        if self.forgetting_factor < 1:
            match self.forget_mode:
                case 'B2P':
                    sigma, mu = self._forget_B2P(key=key_f)
                case 'UI':
                    sigma, mu = self._forget_UI()
        else:
            sigma = self._sigma
            mu = self._mu

        # Predict new sample
        K_dx = (
            self.kernel(self.dictionary, jnp.atleast_2d(x), key=key_dx) *
            self._dictionary_alloc[:, None]
        )
        K_xx = self.kernel(x, key=key_xx) + self._jitter

        _q = self._Q @ K_dx
        projection_uncertainty = K_xx - K_dx.T @ _q
        projection_uncertainty = jnp.where(
            projection_uncertainty < 0,
            0,
            projection_uncertainty,
        )

        _h = sigma @ _q
        noiseless_predictive_var = projection_uncertainty + _q.T @ _h
        noiseless_predictive_var = jnp.where(
            noiseless_predictive_var < 0,
            0,
            noiseless_predictive_var,
        )
        predictive_mean = _q.T @ mu
        predictive_var = self.regularisation + noiseless_predictive_var

        # Update distribution parameters
        _p = jnp.concatenate(
            (_h, jnp.atleast_2d(noiseless_predictive_var)),
            axis=-2,
        )
        mu = (
            jnp.concatenate((mu, predictive_mean), axis=-2) +
            ((y - predictive_mean) / predictive_var) * _p
        )
        sigma = jnp.block([
            [sigma, _h],
            [_h.T, noiseless_predictive_var],
        ]) - (1 / predictive_var) * (_p @ _p.T)

        # Include new sample and add new basis
        _p = jnp.concatenate((_q, jnp.atleast_2d(-1.)))
        Q = jnp.block([
            [self._Q, jnp.zeros((self.dictionary_size, 1))],
            [jnp.zeros((1, self.dictionary_size)), 0],
        ]) + (1 / projection_uncertainty) * (_p @ _p.T)

        # Estimate s02 via maximum likelihood
        _noise_power_numerator = (
            self._noise_power_numerator +
            self.forgetting_factor *
            (y - predictive_mean) ** 2 /
            predictive_var
        )
        _noise_power_denominator = (
            self._noise_power_denominator + self.forgetting_factor
        )

        # Update dictionary
        criterion = jax.lax.cond(
            (projection_uncertainty < self._jitter).squeeze(),
            self._criterion_low_jitter,
            self._criterion_mse_pruning,
            Q,
            mu,
        )

        prune = jnp.argmin(criterion)
        # Condition on the sample to be pruned
        denom = Q[prune, prune]
        denom = jnp.where(
            ~self._x_dictionary_alloc[prune], jnp.inf, denom
        )
        # Something slightly interesting happens when prune == -1 here. As
        # far as the compiler knows, there's a possibility that a single row
        # will be replaced by 2 rows. It looks like the compiler handles
        # this situation by placing the contents of the last replacing row
        # into the replaced index.
        # >>> jnp.arange(16).reshape(4, 4).at[(3, -1), :].set(
        #         jnp.arange(1, 9).reshape(2, 4)
        #     )
        #     Array([[ 0,  1,  2,  3],
        #           [ 4,  5,  6,  7],
        #           [ 8,  9, 10, 11],
        #           [ 5,  6,  7,  8]], dtype=int32)
        # This doesn't bother us here, but we should watch for any unfortunate
        # updates (leading to something like jax deciding to raise an
        # exception)
        _Q = Q.at[(prune, -1), :].set(Q[(-1, prune), :])
        Qs = _Q[:-1, prune]
        Q = _Q.at[:, (prune, -1)].set(_Q[:, (-1, prune)])[..., :-1, :-1]
        Q = Q - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / denom

        # Align the parameter indices correctly
        mu = mu.at[..., (prune, -1), :].set(
            mu[..., (-1, prune), :]
        )[..., :-1, :]
        _sigma = sigma.at[(prune, -1), :].set(sigma[(-1, prune), :])
        sigma = _sigma.at[:, (prune, -1)].set(
            _sigma[:, (-1, prune)]
        )[:-1, :-1]
        dictionary = self.dictionary.at[prune].set(x)
        time_index = self.time_index.at[prune].set(t)
        sigma = (
            (time_index >= 0)[None, :] *
            (time_index >= 0)[:, None] *
            sigma
        )
        mu = jnp.where((time_index < 0)[:, None], 0, mu)

        # Construct the updated kernel
        return KRLST(
            kernel=self.kernel,
            forgetting_factor=self.forgetting_factor,
            regularisation=self.regularisation,
            dictionary_size=self.dictionary_size,
            _mu=mu,
            _sigma=sigma,
            _Q=Q,
            dictionary=dictionary,
            time_index=time_index,
            forget_mode=self.forget_mode,
            _noise_power_numerator=_noise_power_numerator,
            _noise_power_denominator=_noise_power_denominator,
            _jitter=self._jitter,
            nlin=self.nlin,
        )

    def _criterion_low_jitter(
        self,
        Q: Tensor,
        mu: Tensor,
    ):
        return jnp.concatenate(
            (jnp.ones((self.dictionary_size, 1)), jnp.atleast_2d(0.))
        )

    def _criterion_mse_pruning(
        self,
        Q: Tensor,
        mu: Tensor,
    ):
        denom = jnp.diag(Q)
        denom = jnp.where(~self._x_dictionary_alloc, jnp.inf, denom)
        errors = (Q @ mu) / denom[..., None]
        # TODO: We might want reductions other than the sum when determining a
        #       criterion for vectorised kernel machines
        return jnp.abs(errors).sum(-3)

    def predict(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        K_dx = self.kernel(self.dictionary, jnp.atleast_2d(X), key=key)
        mean_est = K_dx.T @ self._Q @ self._mu
        noiseless_var_est = (
            1
            + self._jitter
            + jnp.sum(
                K_dx * ((self._Q @ self._sigma @ self._Q - self._Q) @ K_dx),
                axis=0,
            ).reshape(-1, 1)
        )
        noiseless_var_est = jnp.where(
            noiseless_var_est < 0,
            0,
            noiseless_var_est,
        )
        var_est = (
            self.noise_power_maximum_likelihood *
            (self.regularisation + noiseless_var_est)
        )

        return mean_est, var_est

    def __call__(
        self,
        X: Tensor,
        y: Tensor | None = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor]:
        mean_est, var_est = self.predict(X, key=key)
        if self.nlin is not None:
            #TODO:
            # We're basically assuming this is an error function for now
            mean_est = self.nlin(mean_est.squeeze(), y)
        return mean_est, var_est.squeeze()


# We should double-check the size of the compilation cache. If it's growing,
# we should refactor to a "state"-based formulation.
def observe(
    krlst: KRLST,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> KRLST:
    return krlst.observe(x=x, y=y, t=t, key=key)


def main():
    """
    The example from PyKRLST:
    https://github.com/lckr/PyKRLST/blob/master/PyKRLST_Demo.ipynb
    """
    import jax
    import matplotlib.pyplot as plt
    import numpy as np

    jax.config.update('jax_debug_nans', True)

    def f(x):
        """The function to predict."""
        return x * jnp.sin(x)

    def g(x):
        """Another function to predict."""
        return x * jnp.cos(x)

    # Observations
    X = jnp.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T
    #y = f(X).ravel()
    y = jnp.stack((f(X).ravel(), g(X).ravel()), axis=-1)

    x = jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T

    kernel = RBFKernel(2,)          # Kernel
    M = 5                           # Dictionary budget
    forgetting_factor = 0.999       # Forgetting factor
    regularisation = 1e-5           # Noise-to-signal ratio (used for regulariation)
    mode = "B2P"                    # Forget mode
    krlst = KRLST(kernel=kernel,
                  forgetting_factor=forgetting_factor,
                  regularisation=regularisation,
                  dictionary_size=M,
                  forget_mode=mode)

    # Train in online fashion using at most four basis elements
    krlst = krlst._initialise(X[0], y[0], 0)
    _observe = eqx.filter_jit(observe)
    for t, a, b in zip(jnp.arange(1, 10), X[1:], y[1:]):
        krlst = _observe(krlst, a, b, t)

    # Predict for unknown data
    y_pred, y_std = krlst.predict(x)
    y_pred, y_std = y_pred.squeeze().T, y_std.squeeze().T

    plt.figure(figsize=(10,5))
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(x, g(x), 'g:', label=r'$f(x) = x\,\cos(x)$')
    plt.plot(
        krlst.dictionary.squeeze(),
        krlst.mu.T.squeeze(),
        'k.',
        markersize=20,
        marker="*",
        label="Dictionary Elements",
    )
    plt.plot(X, y, 'r.', markersize=15, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * y_std,
                            (y_pred + 1.9600 * y_std)[::-1]]),
             alpha=.25, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig('/tmp/krlst_result.png')
    breakpoint()


if __name__ == '__main__':
    main()
