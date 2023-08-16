# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopy: community
~~~~~~~~~~~~~~~~~~~~~
Perform community detection on a connectome using a connectopic loss function.
"""
from typing import Tuple

import click
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import pandas as pd

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.init import DirichletInitialiser
from hypercoil.loss import (
    LossArgument,
    LossScheme,
    ModularityLoss,
    UnpackingLossArgument,
)

from hyve import (
    plotdef,
    surf_from_archive,
    surf_scalars_from_cifti,
    scatter_into_parcels,
    plot_to_image,
    save_grid,
)

# TODO: Update this when we choose an actual dataset / update MSC minimal
from hyve_examples import (
    get_schaefer400_cifti,
    get_schaefer400_synthetic_conmat,
)


OUT_ROOT = '/tmp'
NUM_NODES = 400
NUM_COMMUNITIES = 10
COMMUNITY_NU = 1.
GAMMA = 5.
LR = 1e-2
MAX_EPOCH = 500
LOG_INTERVAL = 10
KEY = 0


def community_connectopy(
    *,
    out_root: str = OUT_ROOT,
    num_nodes: int = NUM_NODES,
    num_communities: int = NUM_COMMUNITIES,
    community_nu: float = COMMUNITY_NU,
    gamma: float = GAMMA,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
) -> Tuple[Tensor, Tensor]:
    key_m = jax.random.split(jax.random.PRNGKey(key))[0]
    model = configure_model(num_nodes, num_communities, key=key_m)
    loss = configure_loss(community_nu, gamma)
    model, loss_history = train_model(
        model,
        loss,
        lr,
        max_epoch,
        log_interval,
        key=key,
    )
    plot_f = configure_report(num_communities)
    Q = _to_jax_array(model.Q)
    plot_f(
        template='fsLR',
        load_mask=True,
        parcellation_cifti=get_schaefer400_cifti(),
        model_parcellated=np.array(Q),
        surf_scalars_cmap='Purples',
        surf_scalars_clim=(0.05, 1),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir=out_root,
        window_size=(800, 600),
    )
    pd.DataFrame(loss_history).to_csv(
        f'{out_root}/community_loss_history.tsv',
        sep='\t',
    )
    pd.DataFrame(Q).to_csv(
        f'{out_root}/community_model.tsv',
        sep='\t',
        index=False,
    )


def configure_model(
    num_nodes: int = NUM_NODES,
    num_communities: int = NUM_COMMUNITIES,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    class ConnectopicMaps(eqx.Module):
        Q: Tensor
    Q = jnp.empty((num_nodes, num_communities))
    Q = ConnectopicMaps(Q=Q)
    Q = DirichletInitialiser.init(
        Q,
        axis=-1,
        where='Q',
        concentration=[1],
        num_classes=num_communities,
        key=key,
    )
    return Q


def configure_loss(
    community_nu: float = COMMUNITY_NU,
    gamma: float = GAMMA,
) -> LossScheme:
    return LossScheme((
        ModularityLoss(
            name='CommunityConnectopy',
            nu=community_nu,
            gamma=gamma,
        ),
    ), apply=lambda arg: UnpackingLossArgument(
        Q=arg.Q,
        A=arg.A,
    ))


def configure_optimizer(
    model: eqx.Module,
    lr: float = LR,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    return opt, opt_state


def configure_report(
    num_communities: int = NUM_COMMUNITIES,
) -> callable:
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        scatter_into_parcels('model', 'parcellation'),
        plot_to_image(),
        save_grid(
            n_cols=8,
            n_rows=num_communities,
            padding=10,
            canvas_size=(3200, 300 * num_communities),
            canvas_color=(0, 0, 0),
            fname_spec=(
                f'scalars-communities_count-{num_communities:02d}'
            ),
            sort_by=['surfscalars'],
        ),
    )
    return plot_f


def forward(
    model: eqx.Module,
    loss: LossScheme,
    A: Tensor,
    *,
    key: 'jax.random.PRNGKey',
):
    arg = LossArgument(
        Q=_to_jax_array(model.Q),
        A=A,
    )
    return loss(arg, key=key)


def train_model(
    model: Tensor,
    loss: LossScheme,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    *,
    key: int = KEY,
):
    opt, opt_state = configure_optimizer(model=model, lr=lr)
    key = jax.random.PRNGKey(key)

    # TODO: Fix this.
    A = get_schaefer400_synthetic_conmat()
    A = jnp.asarray(pd.read_csv(A, sep='\t', header=None).values)

    loss_history = []
    for epoch in range(max_epoch):
        key = jax.random.fold_in(key, epoch)
        (loss_value, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward, has_aux=True
        ))(model, loss, A, key=key)
        print(f'[ epoch {epoch} ]')
        print('\n'.join(str((k, v)) for k, v in meta.items()))
        if np.isnan(loss_value):
            raise ValueError('NaN loss')
        loss_history.append(loss_value)
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

        # TODO: We are not reporting at log_interval.

    return model, loss_history


@click.option('-o', '--out-root', default=OUT_ROOT, type=str)
@click.option('-n', '--num-nodes', default=NUM_NODES, type=int)
@click.option('-c', '--num-communities', default=NUM_COMMUNITIES, type=int)
@click.option('--community-nu', default=COMMUNITY_NU, type=float)
@click.option('--gamma', default=GAMMA, type=float)
@click.option('--lr', default=LR, type=float)
@click.option('--max-epoch', default=MAX_EPOCH, type=int)
@click.option('--log-interval', default=LOG_INTERVAL, type=int)
@click.option('--key', default=KEY, type=int)
def main(
    out_root: str = OUT_ROOT,
    num_nodes: int = NUM_NODES,
    num_communities: int = NUM_COMMUNITIES,
    community_nu: float = COMMUNITY_NU,
    gamma: float = GAMMA,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
):
    community_connectopy(
        out_root=out_root,
        num_nodes=num_nodes,
        num_communities=num_communities,
        community_nu=community_nu,
        gamma=gamma,
        lr=lr,
        max_epoch=max_epoch,
        log_interval=log_interval,
        key=key,
    )


if __name__ == "__main__":
    main()
