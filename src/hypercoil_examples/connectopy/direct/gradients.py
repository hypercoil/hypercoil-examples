# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopy: gradients
~~~~~~~~~~~~~~~~~~~~~
Compute gradients of a connectivity matrix using a connectopic loss function.
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
from hypercoil.functional import corr, graph_laplacian, linear_distance
from hypercoil.init import OrthogonalParameter
from hypercoil.loss import (
    LossApply,
    LossArgument,
    LossScheme,
    ConnectopyLoss,
    EigenmapsLoss,
    ConstraintViolationLoss,
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
NUM_CONNECTOPIES = 10
CONNECTOPY_NU = 1.
CONSTRAINT_NU = 100. # severely penalise constraint violations
LR = 1e-2
MAX_EPOCH = 500
LOG_INTERVAL = 10
KEY = 0


def gradients_connectopy(
    *,
    out_root: str = OUT_ROOT,
    num_nodes: int = NUM_NODES,
    num_connectopies: int = NUM_CONNECTOPIES,
    connectopy_nu: float = CONNECTOPY_NU,
    constraint_nu: float = CONSTRAINT_NU,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
) -> Tuple[Tensor, Tensor]:
    key_m = jax.random.split(jax.random.PRNGKey(key))[0]
    model = configure_model(num_nodes, num_connectopies, key=key_m)
    loss = configure_loss(
        connectopy_nu=connectopy_nu,
        constraint_nu=constraint_nu,
        num_connectopies=num_connectopies,
    )
    model, loss_history = train_model(
        model,
        loss,
        lr,
        max_epoch,
        log_interval,
        key=key,
    )
    plot_f = configure_report(num_connectopies=num_connectopies)
    Q = _to_jax_array(model.Q)
    plot_f(
        template='fsLR',
        load_mask=True,
        parcellation_cifti=get_schaefer400_cifti(),
        model_parcellated=np.array(Q),
        surf_scalars_cmap='RdYlBu_r',
        #surf_scalars_clim=(-1, 1),
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
        f'{out_root}/gradients_loss_history.tsv',
        sep='\t',
    )
    pd.DataFrame(Q).to_csv(
        f'{out_root}/gradients_model.tsv',
        sep='\t',
        index=False,
    )


def configure_model(
    num_nodes: int = NUM_NODES,
    num_connectopies: int = NUM_CONNECTOPIES,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    class ConnectopicMaps(eqx.Module):
        Q: Tensor
        theta: Tensor
    key_Q, key_theta = jax.random.split(key)
    Q = jax.random.normal(key_Q, (num_nodes, num_connectopies))
    theta = jnp.flip(
        (0.95 * jax.random.uniform(shape=(10,), key=key_theta) + 0.05).sort()
    )
    Q = ConnectopicMaps(Q=Q, theta=theta)
    Q = OrthogonalParameter.map(model=Q, where='Q')
    return Q


def configure_loss(
    connectopy_nu: float = CONNECTOPY_NU,
    constraint_nu: float = CONSTRAINT_NU,
    num_connectopies: int = NUM_CONNECTOPIES,
) -> LossScheme:
    descending_constraint = (
        -jnp.eye(num_connectopies) + jnp.eye(num_connectopies, k=1)
    )
    return LossScheme((
        LossApply(
            EigenmapsLoss(
                name='GradientConnectopy',
                nu=connectopy_nu,
            ),
            apply=lambda arg: UnpackingLossArgument(
                Q=arg.Q,
                A=arg.A,
                theta=arg.theta,
            ),
        ),
        LossApply(
            ConstraintViolationLoss(
                name='EigenvalueConstraints',
                nu=constraint_nu,
                constraints=(
                    lambda x, key: jax.nn.relu(x - 1), # Bound eigenvalues below 1
                    lambda x, key: jax.nn.relu(0.05 - x), # Bound eigenvalues above 0.05
                    lambda x, key: descending_constraint @ x, # Ensure descending eigenvalues
                ),
            ),
            apply=lambda arg: arg.theta,
        )
    ))


def configure_optimizer(
    model: eqx.Module,
    lr: float = LR,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    return opt, opt_state


def configure_report(
    num_connectopies: int = NUM_CONNECTOPIES,
) -> callable:
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        scatter_into_parcels('model', 'parcellation'),
        plot_to_image(),
        save_grid(
            n_cols=8,
            n_rows=num_connectopies,
            padding=10,
            canvas_size=(3200, 300 * num_connectopies),
            canvas_color=(0, 0, 0),
            fname_spec=(
                f'scalars-gradients_count-{num_connectopies:02d}'
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
        theta=_to_jax_array(model.theta),
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
@click.option('-c', '--num-connectopies', default=NUM_CONNECTOPIES, type=int)
@click.option('--connectopy-nu', default=CONNECTOPY_NU, type=float)
@click.option('--constraint-nu', default=CONSTRAINT_NU, type=float)
@click.option('--lr', default=LR, type=float)
@click.option('--max-epoch', default=MAX_EPOCH, type=int)
@click.option('--log-interval', default=LOG_INTERVAL, type=int)
@click.option('--key', default=KEY, type=int)
def main(
    out_root: str = OUT_ROOT,
    num_nodes: int = NUM_NODES,
    num_connectopies: int = NUM_CONNECTOPIES,
    connectopy_nu: float = CONNECTOPY_NU,
    constraint_nu: float = CONSTRAINT_NU,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
):
    gradients_connectopy(
        out_root=out_root,
        num_nodes=num_nodes,
        num_connectopies=num_connectopies,
        connectopy_nu=connectopy_nu,
        constraint_nu=constraint_nu,
        lr=lr,
        max_epoch=max_epoch,
        log_interval=log_interval,
        key=key,
    )


if __name__ == "__main__":
    main()
