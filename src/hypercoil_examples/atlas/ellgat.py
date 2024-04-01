# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ELL-format graph attention network (GAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a simple graph attention module that operates on an adjacency index
that corresponds to an ELL-format sparse matrix. It uses the "GATv2"
formulation from the paper "How Attentive are Graph Attention Networks?" by
Brody et al. We use this as the base module for a u-net style architecture
that operates on data defined over icosphere meshes.
"""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.engine import Tensor


class ELLGAT(eqx.Module):
    """
    Graph attention network (GAT) for ELL-format data.

    This is a simple graph attention network that operates on ELL-format data.
    """
    query_features: int
    out_features: int
    attn_heads: int
    query_weight: Tensor
    key_weight: Tensor
    attn_weight: Tensor
    nlin: callable = jax.nn.leaky_relu
    key_features: Optional[int] = None

    def __init__(
        self,
        query_features: int,
        out_features: int,
        attn_heads: int = 1,
        nlin: callable = jax.nn.leaky_relu,
        key_features: Optional[int] = None,
        *,
        key: 'jax.random.PRNGKey',
    ):
        if key_features is None:
            key_features = query_features
        self.query_features = query_features
        self.key_features = key_features
        self.out_features = out_features
        self.attn_heads = attn_heads
        key_attn, key_qweight, key_kweight = jax.random.split(key, 3)
        qw_scale = 2 * jnp.sqrt(6.0 / (query_features + out_features * attn_heads))
        kw_scale = 2 * jnp.sqrt(6.0 / (key_features + out_features * attn_heads))
        a_scale = 2 * jnp.sqrt(6.0 / (out_features + attn_heads))
        self.key_weight = (jax.random.uniform(
            key_kweight,
            (attn_heads, out_features, key_features),
        ) - 0.5) * kw_scale
        self.query_weight = (jax.random.uniform(
            key_qweight,
            (attn_heads, out_features, query_features),
        ) - 0.5) * qw_scale
        self.attn_weight = (jax.random.uniform(
            key_attn,
            (attn_heads, out_features),
        ) - 0.5) * a_scale
        self.nlin = nlin

    def __call__(
        self,
        adj: Tensor,
        Q: Tensor,
        K: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if K is None:
            K = Q
        K = jnp.einsum('...hoi,...in->...hon', self.key_weight, K)
        # X = self.nlin(
        #     jnp.einsum(
        #         '...hoi,...in,...honk->...honk',
        #         self.query_weight, Q, K[..., adj],
        #     )
        # )
        Q = jnp.einsum('...hoi,...in->...hon', self.query_weight, Q)
        X = self.nlin(
            jnp.einsum('...hon,...honk->...honk', Q, K[..., adj])
        )
        attn = jax.nn.softmax(jnp.where(adj == -1, -jnp.inf, X), axis=-1)
        attn = jnp.where(jnp.isnan(attn), 0, attn)
        # attn = jnp.einsum(
        #     '...hwnk,...hw->...hnk',
        #     attn,
        #     self.attn_weight,
        # )
        # return jnp.einsum(
        #     '...hnk,...honk->...hon',
        #     attn,
        #     jnp.where(adj == -1, 0, X),
        # )
        return jnp.einsum(
            '...hwnk,...hw,...honk->...hon',
            attn,
            self.attn_weight,
            jnp.where(adj == -1, 0, X),
        )


class UnitSphereNorm(eqx.Module):
    """
    A simple normalisation layer that normalises the input to the unit sphere.
    """
    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        norm = jnp.linalg.norm(X, axis=-2, keepdims=True)
        return jnp.where(norm == 0, X, X / norm)


class ELLGATBlock(eqx.Module):
    layers: Tuple[ELLGAT, ELLGAT]
    nlin: callable = jax.nn.leaky_relu
    norm: Optional[eqx.Module] = None

    def __init__(
        self,
        query_features: int,
        out_features: int,
        attn_heads: int = 1,
        nlin: callable = jax.nn.leaky_relu,
        norm: Optional[eqx.Module] = None,
        key_features: Optional[Union[int, Tuple[int, int]]] = None,
        *,
        key: 'jax.random.PRNGKey',
    ):
        key1, key2 = jax.random.split(key)
        if not isinstance(key_features, tuple):
            key_features = (key_features, key_features)
        self.layers = (
            ELLGAT(
                query_features=query_features,
                key_features=key_features[0],
                out_features=out_features,
                attn_heads=attn_heads,
                nlin=nlin,
                key=key1,
            ),
            ELLGAT(
                query_features=out_features * attn_heads,
                key_features=key_features[1],
                out_features=out_features,
                attn_heads=attn_heads,
                nlin=nlin,
                key=key2,
            ),
        )
        self.nlin = nlin
        self.norm = norm

    def __call__(
        self,
        adj: Tensor,
        Q: Tensor,
        K: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if key is None:
            key1, key2 = None, None
        else:
            key1, key2 = jax.random.split(key)
        if isinstance(K, tuple):
            K1, K2 = K
        else:
            K1 = K2 = K
        Q = self.layers[0](adj, Q, K1, key=key1)
        # Collapse the head and feature dimensions
        Q = self.nlin(Q).reshape(
            *Q.shape[:-3],
            Q.shape[-3] * Q.shape[-2],
            Q.shape[-1],
        )
        if self.norm is not None:
            Q = self.norm(Q)
        Q = self.layers[1](adj, Q, K2, key=key2)
        # Collapse the head and feature dimensions
        return Q.reshape(
            *Q.shape[:-3],
            Q.shape[-3] * Q.shape[-2],
            Q.shape[-1],
        )


def main():
    # Test the ELLGAT module
    from hypercoil_examples.atlas.icosphere import (
        icosphere, connectivity_matrix
    )
    vertices, faces = icosphere(60)
    edges = connectivity_matrix(vertices, faces)
    model = ELLGAT(3, 4, 2, key=jax.random.PRNGKey(0))
    Q = jax.random.normal(
        jax.random.PRNGKey(17),
        (model.key_features, vertices.shape[0]),
    )
    out = model(edges, Q)
    out = out.reshape(
        *out.shape[:-3],
        out.shape[-3] * out.shape[-2],
        out.shape[-1],
    )
    assert 0


if __name__ == '__main__':
    main()
