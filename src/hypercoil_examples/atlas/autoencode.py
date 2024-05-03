# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ELLGAT autoencoder
~~~~~~~~~~~~~~~~~~
Test the ELLGAT implementation using a simple autoencoding problem. We learn
to approximately reconstruct the input--the eigenmode maps from Pang et al.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import pandas as pd

from hypercoil.engine import Tensor
from hypercoil_examples.atlas.ellgat import ELLGATBlock, ELLGAT
from hypercoil_examples.atlas.positional import EIGENMODES_PATH
from hypercoil_examples.atlas.unet import get_base_coor_mask_adj

MAX_EPOCH = 100
LEARNING_RATE = 0.001


class TestModel(eqx.Module):
    block: ELLGATBlock
    readout: ELLGAT

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        readout_dim: int,
        attn_heads: int = 1,
        nlin: callable = jax.nn.leaky_relu,
        norm: Optional[callable] = None,
        dropout: Optional[float] = None,
        dropout_inference: bool = False,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        key = jax.random.split(key, 2)
        self.block = ELLGATBlock(
            query_features=in_dim,
            out_features=hidden_dim,
            attn_heads=attn_heads,
            nlin=nlin,
            norm=norm,
            dropout=dropout,
            dropout_inference=dropout_inference,
            key=key[0],
        )
        self.readout = ELLGAT(
            query_features=hidden_dim * attn_heads,
            out_features=readout_dim,
            attn_heads=1,
            nlin=nlin,
            dropout=dropout,
            dropout_inference=dropout_inference,
            key=key[1],
        )

    def __call__(
        self,
        adj,
        X,
        *,
        inference: bool = False,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        key = jax.random.split(key, 2)
        X = self.block(adj=adj, Q=X, key=key[0], inference=inference)
        X = self.readout(adj=adj, Q=X, key=key[1], inference=inference)
        return X


def forward(
    model: TestModel,
    *,
    adj: Tensor,
    data: Tensor,
    key: 'jax.random.PRNGKey',
):
    result = model(adj, data, key=key)[0]
    return ((result[0] - data) ** 2).sum()


def main():
    arrays = {
        k: pd.read_csv(v, sep=' ', header=None).values
        for k, v in EIGENMODES_PATH.items()
    }
    data_L = arrays['cortex_L']
    data_R = arrays['cortex_R']
    model = TestModel(
        in_dim=200,
        hidden_dim=64,
        readout_dim=200,
        attn_heads=4,
        dropout=0.1,
        key=jax.random.PRNGKey(0),
    )
    _, mask_L, adj_L = get_base_coor_mask_adj('L')
    _, mask_R, adj_R = get_base_coor_mask_adj('R')
    adj_L = jnp.concatenate(
        (jnp.arange(len(adj_L))[..., None], adj_L),
        axis=-1,
    )
    adj_R = jnp.concatenate(
        (jnp.arange(len(adj_R))[..., None], adj_R),
        axis=-1,
    )
    #adj_L = adj_L[mask_L]
    #adj_R = adj_R[mask_R]
    #data_L = data_L[mask_L]
    #data_R = data_R[mask_R]
    opt = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []
    for i in range(MAX_EPOCH):
        key = jax.random.PRNGKey(i)
        key_L, key_R = jax.random.split(key)
        loss_epoch = 0
        for adj, data, key in (
            (adj_L, data_L, key_L),
            (adj_R, data_R, key_R),
        ):
            loss, grad = eqx.filter_value_and_grad(
                eqx.filter_jit(forward)
            )(
            # forward(
                model,
                adj=adj,
                data=data.T,
                key=key,
            )
            updates, opt_state = opt.update(
                eqx.filter(grad, eqx.is_inexact_array),
                opt_state,
                eqx.filter(model, eqx.is_inexact_array),
            )
            loss_epoch += loss.item()
        model = eqx.apply_updates(model, updates)
        losses += [loss_epoch]
        print(f'Epoch {i}: loss {losses[-1]}')
    result_L = model(
        adj_L, data_L.T, key=jax.random.PRNGKey(0), inference=True
    )
    result_R = model(
        adj_R, data_R.T, key=jax.random.PRNGKey(0), inference=True
    )
    plt.plot(losses)
    plt.savefig('/tmp/losses.png')
    plt.close()
    jnp.save('/tmp/eigenmodes_result_L.npy', result_L, allow_pickle=False)
    jnp.save('/tmp/eigenmodes_result_R.npy', result_R, allow_pickle=False)
    assert 0


if __name__ == '__main__':
    main()
