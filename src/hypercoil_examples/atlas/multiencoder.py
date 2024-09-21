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
from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.nn.atlas import AtlasLinear
from hypercoil.engine import Tensor


class MultiEncoder(eqx.Module):
    encoders: Sequence[eqx.Module]
    mask: Tensor
    reduced_mask: Tensor
    reduced_slices: Sequence[Tuple[int, int]]
    scales: Sequence[float]
    transforms: Sequence[callable]
    compartments: Sequence[str]
    compartment_masks: Sequence[Tensor]
    compartment_slices: Optional[Sequence[Tuple[int, int]]]
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
        self.reduced_mask = self.mask[self.mask.any(-1)]
        self.reduced_slices = [
            tuple(e) for e in zip(
                self.reduced_mask.argmax(0).tolist(),
                self.reduced_mask.sum(0).tolist(),
            )
        ]
        self.scales = scales
        self.transforms = transforms
        self.compartments = compartments
        self.compartment_masks = compartment_masks
        # Try to build compartment slices--if we can, then the forward pass
        # of the encoder will be JIT-able
        compartment_slices = [False for _ in compartment_masks]
        for i, cmask in enumerate(compartment_masks):
            if cmask.all():
                compartment_slices[i] = None
            else:
                indices = jnp.where(cmask)[0]
                start = indices.min()
                end = indices.max() + 1
                if jnp.all(cmask[start:end]):
                    compartment_slices[i] = (start.item(), end.item())
        if all([cs is not False for cs in compartment_slices]):
            self.compartment_slices = tuple(compartment_slices)
        else:
            self.compartment_slices = None
        self.input_dim = input_dim
        self.concat = concat

    def rescale(
        self,
        X: Tensor,
        use_reduced_slices: bool = True,
        use_reduced_mask: bool = False,
        concatenate: bool = True,
    ) -> Tensor:
        masks = self.reduced_mask if use_reduced_mask else self.mask
        if use_reduced_slices:
            X = [
                jax.lax.dynamic_slice(
                    X,
                    (0,) * (X.ndim - 1) + (s[0],),
                    X.shape[:-1] + (s[1],),
                )
                for s in self.reduced_slices
            ]
            X = [
                enc /
                jnp.linalg.norm(enc, axis=-1, keepdims=True) *
                jnp.sqrt(scale)
                for enc, scale in zip(X, self.scales)
            ]
            if concatenate:
                X = jnp.concatenate(X, axis=-1)
            return X
        # Doesn't work with jit
        if not concatenate:
            X = [X[..., mask] for mask in masks.T]
            return [
                enc /
                jnp.linalg.norm(enc, axis=-1, keepdims=True) *
                jnp.sqrt(scale)
                for enc, scale in zip(X, self.scales)
            ]
        for mask, scale in zip(masks.T, self.scales):
            X = jnp.where(
                jnp.expand_dims(mask, axis=tuple(range(X.ndim - 1))),
                (
                    X / jnp.linalg.norm(X * mask, axis=-1, keepdims=True) *
                    jnp.sqrt(scale)
                ),
                X,
            )
        return X

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
        if self.compartment_slices is not None:
            encoded = [
                enc if s is None else jax.lax.dynamic_slice(
                    enc,
                    (0,) * (enc.ndim - 1) + (s[0],),
                    enc.shape[:-1] + (s[1],),
                )
                for enc, s in zip(encoded, self.compartment_slices)
            ]
        else:
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
                jax.lax.dynamic_slice(
                    encoded[0],
                    (0,) * (encoded[0].ndim - 1) + (s[0],),
                    encoded[0].shape[:-1] + (s[1],),
                )
                for s in self.reduced_slices
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


def configure_multiencoder(
    use_icosphere: bool = True,
    use_consensus: bool = True,
    use_7net: bool = True,
    scales: Union[Sequence[float], Literal['auto']] = 'auto',
):
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
        create_icosphere_encoder() if use_icosphere else None,
        create_consensus_encoder() if use_consensus else None,
        create_7net_encoder() if use_7net else None,
    )
    encoders = tuple(e for e in encoders if e is not None)
    if scales == 'auto':
        scales = (1 / len(encoders),) * len(encoders)
    def transform(enc: Tensor) -> Tensor:
        loc = enc
        scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
        return logistic_mixture_threshold(
            incomplete_mahalanobis_transform(loc)[0],
            #jnp.arctanh(jnp.where(jnp.isclose(loc, 1.), 1 - 1e-7, loc)),
            scale,
            k=0.9,
            axis=-1,
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
    multiencoder = configure_multiencoder(use_7net=False)
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
        [
            (1 / len(multiencoder.encoders)) / jnp.sqrt(i)
            for i in multiencoder.mask.sum(0)
        ]
    )
    assert (
        jnp.abs(tst[..., None] - means).argmin(0) == jnp.arange(len(means))
    ).all()
    sizes = [0] + jnp.cumsum(multiencoder.reduced_mask.sum(0)).tolist()
    assert all([
        jnp.allclose(
            jnp.linalg.norm(X[..., start:end], axis=-1) ** 2,
            1 / len(multiencoder.encoders)
        )
        for start, end in zip(sizes[:-1], sizes[1:])
    ])
    for m, t in zip(means, tst):
        assert m > t
    assert 0


if __name__ == '__main__':
    main()
