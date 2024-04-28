# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Training loop
~~~~~~~~~~~~~
Training loop for the parcellation model
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from hypercoil_examples.atlas.aligned_dccc import (
    get_msc_dataset, _get_data
)
from hypercoil_examples.atlas.cross2subj import visualise
from hypercoil_examples.atlas.model import (
    init_full_model,
    forward,
)
from hypercoil_examples.atlas.positional import (
    get_coors
)

LEARNING_RATE = 0.001
REPORT_INTERVAL = 10


def main(subject: str = '01', session: str = '01', num_parcels: int = 100):
    coor_L, coor_R = get_coors()
    T = _get_data(get_msc_dataset(subject, session))
    model, encoder, template = init_full_model(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        num_parcels=num_parcels,
    )
    opt = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    encoder_result = encoder(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        M=template,
    )
    losses = []
    for i in range(2000):
        print(i)
        (loss, meta), grad = eqx.filter_value_and_grad(
            eqx.filter_jit(forward),
            has_aux=True,
        )(
            model,
            coor={
                'cortex_L': coor_L,
                'cortex_R': coor_R,
            },
            encoder_result=encoder_result,
            encoder=encoder,
            compartment='cortex_L',
            key=jax.random.PRNGKey(0),
        )
        if jnp.isnan(loss):
            print(f'NaN loss at epoch {i}. Skipping update')
            continue
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        losses += [loss.item()]
        del updates
        if i % REPORT_INTERVAL == 0:
            print('\n'.join([f'[]{k}: {v}' for k, v in meta.items()]))
            visualise(
                name=f'MRF_epoch-{i}',
                log_prob_L=model.regulariser[
                    'cortex_L'
                ].selectivity_distribution.log_prob(
                    template['cortex_L']
                ) + model.regulariser[
                    'cortex_L'
                ].spatial_distribution.log_prob(
                    coor_L
                ),
                log_prob_R=model.regulariser[
                    'cortex_R'
                ].selectivity_distribution.log_prob(
                    template['cortex_R']
                ) + model.regulariser[
                    'cortex_R'
                ].spatial_distribution.log_prob(
                    coor_R
                ),
            )
            #TODO: Load a specific set of subjects and sessions
            # P, _, _ = model(
            #     coor={
            #         'cortex_L': coor_L,
            #         'cortex_R': coor_R,
            #     },
            #     encoder=encoder,
            #     encoder_result=encoder_result,
            #     compartments=('cortex_L', 'cortex_R'),
            #     key=jax.random.PRNGKey(0),
            # )
            # visualise(
            #     name=f'epoch_{i}',
            #     log_prob_L=P['cortex_L'],
            #     log_prob_R=P['cortex_R'],
            # )
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('/tmp/losses.png')
    assert 0


if __name__ == '__main__':
    main()
