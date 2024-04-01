# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multi-encoder
~~~~~~~~~~~~~
Multiple encoders are used to encode the input data into different
representations. Each encoder is initialised to map a different scale of
selectivity space.
"""
from typing import Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.nn.atlas import AtlasLinear
from hypercoil.engine import Tensor


class MultiEncoder(eqx.Module):
    encoders: Sequence[eqx.Module]
    mask: Tensor
    scales: Sequence[float]
    transforms: Sequence[callable]
    compartments: Sequence[str]
    compartment_masks: Sequence[Tensor]
    input_dim: int
    concat: Optional[Literal['pretransform', 'posttransform']]

    def __init__(
        self,
        encoders: Sequence[eqx.Module],
        scales: Sequence[float],
        transforms: Sequence[callable],
        compartments: Sequence[str],
        concat: Optional[Literal['pretransform', 'posttransform']] = None,
        *,
        key: 'jax.random.PRNGKey',
    ):
        input_dim = sum([
            encoders[0].maps[c].shape[1] for c in compartments
        ])
        assert all(
            input_dim == sum([e.maps[c].shape[1] for c in compartments])
            for e in encoders
        )
        self.encoders = tuple(
            AtlasLinear.from_atlas(enc, encode=True, key=key)
            for enc in encoders
        )
        compartment_masks = [
            jnp.asarray(
                sum([
                    e.weight[c].shape[0] *
                    ([True] if c in compartments else [False])
                    for c in e.weight
                ], [])
            )
            for e in self.encoders
        ]
        encoder_masks = jnp.asarray(
            sum(
                [[i] * len(e) for i, e in enumerate(compartment_masks)],
                [],
            )
        )
        mask = jnp.eye(len(encoders), dtype=bool)[encoder_masks]
        for i, cmask in enumerate(compartment_masks):
            mask = mask.at[mask[..., i], i].set(cmask)
        self.mask = mask
        self.scales = scales
        self.transforms = transforms
        self.compartments = compartments
        self.compartment_masks = compartment_masks
        self.input_dim = input_dim
        self.concat = concat

    @property
    def reduced_mask(self) -> Tensor:
        return self.mask[self.mask.any(-1)]

    def rescale(
        self,
        X: Tensor,
        use_reduced_mask: bool = True,
        concatenate: bool = True,
    ) -> Tensor:
        masks = self.reduced_mask if use_reduced_mask else self.mask
        X = [X[..., mask] for mask in masks.T]
        X = [
            enc /
            jnp.linalg.norm(enc, axis=-1, keepdims=True) *
            jnp.sqrt(scale)
            for enc, scale in zip(X, self.scales)
        ]
        if not concatenate:
            return X
        return jnp.concatenate(X, axis=-1)

    def __call__(
        self, X: Tensor, *, key: Optional['jax.random.PRNGKey'] = None
    ) -> Tensor:
        encoded = [model(X) for model in self.encoders]
        encoded = [
            jnp.concatenate(
                tuple(enc[c] for c in self.compartments)
            )
            for enc in encoded
        ]
        encoded = [
            enc[..., mask] for enc, mask in zip(
                encoded,
                self.compartment_masks,
            )
        ]
        if self.concat == 'pretransform':
            encoded = [jnp.concatenate(encoded, axis=-1)]
        for transform in self.transforms:
            encoded = [transform(enc) for enc in encoded]
        if len(encoded) == 1:
            concat = True
            encoded = [
                encoded[0][..., self.mask[self.mask.any(-1), i]]
                for i in range(len(self.encoders))
            ]
        else:
            concat = False
        encoded = [
            enc /
            jnp.linalg.norm(enc, axis=-1, keepdims=True) *
            jnp.sqrt(scale)
            for enc, scale in zip(encoded, self.scales)
        ]
        if concat or self.concat == 'posttransform':
            encoded = jnp.concatenate(encoded, axis=-1)
        return encoded


def configure_multiencoder():
    from hypercoil_examples.atlas.encoders import (
        create_icosphere_encoder,
        create_consensus_encoder,
        create_7net_encoder,
    )
    from hypercoil_examples.atlas.selectransform import (
        incomplete_mahalanobis_transform,
        logistic_mixture_threshold,
    )
    encoders = (
        create_icosphere_encoder(),
        create_consensus_encoder(),
        create_7net_encoder(),
    )
    scales = (1 / 3, 1 / 3, 1 / 3)
    def transform(enc: Tensor) -> Tensor:
        loc = enc
        scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
        return logistic_mixture_threshold(
            incomplete_mahalanobis_transform(loc)[0], scale, k=0.9, axis=-1
        )
    multiencoder = MultiEncoder(
        encoders=encoders,
        scales=scales,
        transforms=(transform,),
        compartments=('cortex_L', 'cortex_R'),
        concat='pretransform',
        key=jax.random.PRNGKey(33),
    )
    return multiencoder


def main():
    multiencoder = configure_multiencoder()
    X = jax.random.uniform(
        key=jax.random.PRNGKey(0),
        shape=(multiencoder.input_dim, 100),
    )
    X = multiencoder(X)
    means = jnp.asarray([
        jnp.abs(X).mean(0)[mask].mean()
        for mask in multiencoder.mask[multiencoder.mask.any(-1)].T
    ])
    tst = jnp.asarray(
        [(1 / 3) / jnp.sqrt(i) for i in multiencoder.mask.sum(0)]
    )
    assert (
        jnp.abs(tst[..., None] - means).argmin(0) == jnp.arange(len(means))
    ).all()
    for m, t in zip(means, tst):
        assert m > t
    assert 0


if __name__ == '__main__':
    main()
