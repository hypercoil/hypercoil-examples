# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data dimension
~~~~~~~~~~~~~~
Adaptive sphering and generalised whitening
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from hypercoil.engine import Tensor

from hypercoil_examples.atlas.vmf import generalised_whitening


COMPONENT = 15
N_POINTS = 10000
DATA_DIM = 25
MAX_EPOCH = 4000


class AdaptiveSpheringSigmoid(eqx.Module):
    inflection: Tensor
    lim: float

    def sphering(self, shape: int):
        return (
            1 - jax.nn.sigmoid(
                jnp.linspace(-self.lim, self.lim, shape)
                - self.lim * (2 * self.inflection - 1.0)
            )
        )

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(X, sphering=self.sphering(shape))


class AdaptiveSpheringSigmoidPlus(eqx.Module):
    inflection: Tensor
    lim: float

    def sphering(self, shape: int):
        sig = (
            1 - jax.nn.sigmoid(
                jnp.linspace(-self.lim, self.lim, shape)
                - self.lim * (2 * self.inflection - 1.0)
            )
        )
        return jnp.where(jnp.arange(shape) < 1, 1., sig)

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(X, sphering=self.sphering(shape))


class AdaptiveSpheringHalfSigmoid(eqx.Module):
    inflection: Tensor
    lim: float

    def sphering(self, shape: int):
        sig = (
            1 - jax.nn.sigmoid(
                jnp.linspace(-self.lim, self.lim, shape)
                - self.lim * (2 * self.inflection - 1.0)
            )
        )
        return jnp.where(sig < 0.5, 2 * sig, 1.)

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(X, sphering=self.sphering(shape))


class AdaptiveSpheringHalfArctangent(eqx.Module):
    inflection: Tensor
    lim: float
    floor: float = 0.

    def sphering(self, shape: int):
        arg = (
            jnp.linspace(-self.lim, self.lim, shape)
            - self.lim * (2 * self.inflection - 1.0)
        )
        sig = (
            0.5 - jnp.arctan(jnp.pi * arg / 2) / jnp.pi
        )
        return (1 - self.floor) * jnp.where(sig < 0.5, 2 * sig, 1.) + self.floor

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(X, sphering=self.sphering(shape))


class AdaptiveSpheringArctangent(eqx.Module):
    inflection: Tensor
    lim: float
    floor: float = 0.

    def sphering(self, shape: int):
        arg = (
            jnp.linspace(-self.lim, self.lim, shape)
            - self.lim * (2 * self.inflection - 1.0)
        )
        val =  (
            0.5 - jnp.arctan(jnp.pi * arg / 2) / jnp.pi
        )
        return (1 - self.floor) * jnp.where(jnp.arange(shape) < 1, 1., val) + self.floor

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(X, sphering=self.sphering(shape))


class AdaptiveSphering(eqx.Module):
    S: Tensor

    def sphering(self, shape: int):
        return self.S

    def __call__(self, X: Tensor) -> Tensor:
        shape = X.shape[-1]
        return generalised_whitening(
            X, sphering=jax.nn.relu(self.sphering(shape))
        )


class Model(eqx.Module):
    linear: Tensor
    adasphere: AdaptiveSphering
    
    def __call__(self, X: Tensor):
        W = self.adasphere(X)
        v = self.linear / jnp.linalg.norm(self.linear, axis=-1, keepdims=True)
        return X @ W.T @ v.T


def forward(model, X, Y):
    Y_hat = model(X.T).squeeze()
    # We don't care about the sign of the correlation
    return jnp.minimum(
        ((Y - Y_hat) ** 2),
        ((Y + Y_hat) ** 2),
    ).mean()


def main():
    data = jnp.stack([
        2 ** -i * jax.random.normal(jax.random.PRNGKey(i), shape=(N_POINTS,))
        for i in range(DATA_DIM)
    ])
    A = jax.random.normal(jax.random.PRNGKey(99), (DATA_DIM, DATA_DIM))
    _, _, V = jnp.linalg.svd(data, full_matrices=False)
    Y = V[COMPONENT]
    Y = Y / jnp.std(Y)
    #Q, _ = jnp.linalg.qr(A)
    #data = Q @ data

    #adasphere = AdaptiveSphering(S=1 - jnp.arange(25) / 25)
    adasphere = AdaptiveSpheringSigmoid(inflection=jnp.asarray([1.]), lim=100.)
    linear = jax.random.normal(jax.random.PRNGKey(47), (1, 25))
    model = Model(linear, adasphere)

    optim = optax.adamw(1e-3, weight_decay=5e-1)
    optim_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []
    inflections = []
    for e in range(MAX_EPOCH):
        loss, grad = eqx.filter_jit(eqx.filter_value_and_grad(forward))(
            model, data, Y
        )
        losses += [loss]
        inflections += [model.adasphere.inflection.squeeze()]
        updates, optim_state = optim.update(
            eqx.filter(grad, eqx.is_inexact_array),
            optim_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        print(
            f'Epoch {e}',
            loss,
            model.linear.squeeze()[COMPONENT],
            model.adasphere.inflection.squeeze(),
        )

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(jnp.linspace(0, 25, 1000), model.adasphere.sphering(1000))
    plt.axvline(COMPONENT, c='orange', ls=':')
    plt.subplot(3, 2, 2)
    plt.plot(model.linear.squeeze())
    plt.axvline(COMPONENT, c='orange', ls=':')
    plt.subplot(3, 2, 3)
    plt.scatter(Y, model(data.T))
    plt.subplot(3, 2, 5)
    plt.plot(losses)
    plt.subplot(3, 2, 6)
    plt.plot(inflections)
    plt.savefig(f'/tmp/sphering{COMPONENT}.png')
    assert 0


if __name__ == '__main__':
    main()
