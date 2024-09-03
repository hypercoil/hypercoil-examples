# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Simple error
~~~~~~~~~~~~
An equinox module that computes a simple error for a vector of predictions and
a vector of targets. The prediction vector can be split along any single axis
and each shard annotated as either categorical (in which case the error is
a multinomial cross entropy) or continuous (in which case the error is a mean
squared error).
"""
from typing import Sequence, Optional

import jax.numpy as jnp
import equinox as eqx
from optax import (
    softmax_cross_entropy,
    squared_error,
)

from hypercoil.engine import Tensor


def simple_error(
    predictions: Tensor,
    targets: Tensor,
    split_indices: Sequence[int],
    shard_categorical: Sequence[bool],
    split_axis: int = -1,
    categorical_predictions_multiplier: float = 1.,
) -> Tensor:
    """
    Compute the error between predictions and targets, where the predictions
    are split along the `split_axis` and each shard is annotated as either
    categorical or continuous.

    Parameters
    ----------
    predictions : Tensor
        The predictions.
    targets : Tensor
        The targets.
    split_indices : Sequence[int]
        The indices at which to split the predictions.
    shard_categorical : Sequence[bool]
        Whether each shard is categorical.
    split_axis : int, default=-1
        The axis along which to split the predictions.
    categorical_predictions_multiplier : float, default=1.
        The multiplier to apply to categorical predictions.

    Returns
    -------
    Tensor
        The error.
    """
    # Split the predictions and targets.
    predictions = jnp.split(predictions, split_indices, axis=split_axis)
    targets = jnp.split(targets, split_indices, axis=split_axis)

    # Compute the error for each shard.
    errors = jnp.concatenate(
        [
            softmax_cross_entropy(
                categorical_predictions_multiplier * p.swapaxes(split_axis, -1),
                t.swapaxes(split_axis, -1),
            )[..., None].swapaxes(split_axis, -1)
            if c
            # else squared_error(p, t).mean(axis=split_axis, keepdims=True)
            else squared_error(p, t)
            for p, t, c in zip(predictions, targets, shard_categorical)
        ],
        axis=split_axis,
    )
    return errors


class SimpleError(eqx.Module):
    split_indices: Sequence[int]
    shard_categorical: Sequence[bool]
    split_axis: int
    categorical_predictions_multiplier: float

    def __call__(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return simple_error(
            predictions,
            targets,
            self.split_indices,
            self.shard_categorical,
            self.split_axis,
            self.categorical_predictions_multiplier,
        )


def main():
    err = SimpleError(
        split_indices=(5,),
        shard_categorical=(False, True),
        split_axis=-2,
        categorical_predictions_multiplier=1.,
    )
    X = jnp.concatenate(
        (
            jnp.arange(125).reshape(5, 5, 5) - 62.,
            jnp.tile(jnp.eye(5)[..., None], (1, 1, 5)),
        ),
        axis=-2,
    )
    Y = jnp.concatenate(
        (
            jnp.arange(125).reshape(5, 5, 5) - 63.,
            jnp.tile(jnp.eye(5)[..., None], (1, 1, 5)),
        ),
        axis=-2,
    )
    e = err(X, Y)
    assert e.shape == (5, 6, 5)
    assert jnp.all(e.sum((0, -1))[:5] == 25)
    assert jnp.allclose(e[:, -1, :].mean(), e[:, -1, :])
    breakpoint()


if __name__ == '__main__':
    main()