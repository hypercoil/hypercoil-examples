# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Template construction
~~~~~~~~~~~~~~~~~~~~~
Training loop for template construction
"""
import logging
import pickle
from itertools import product
from typing import Mapping, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from hypercoil.engine import _to_jax_array
from hypercoil_examples.atlas.const import HCP_DATA_SPLIT_DEF_ROOT
from hypercoil_examples.atlas.data import (
    get_hcp_dataset, get_msc_dataset, _get_data
)
from hypercoil_examples.atlas.model import init_encoder_model
from hypercoil_examples.atlas.positional import get_coors

MAX_EPOCH = 14785 # 9801
CHECKPOINT_INTERVAL = 100 # 1848 # 1960 #
MSC_SUBJECTS = ('01', '02', '03') #, '04', '05', '06', '07', '08', '09', '10',)
MSC_SESSIONS = ('01', '02', '03') #, '04', '05', '06', '07', '08', '09', '10',)
MSC_SUBJECTS = ('05', '06')
MSC_TASKS = (
    'rest', 'motor_run-01', 'motor_run-02',
    'glasslexical_run-01', 'glasslexical_run-02',
    'memoryfaces', 'memoryscenes', 'memorywords',
)
HCP_TASKS = (
    'REST1', 'REST2', 'EMOTION', 'GAMBLING',
    'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM',
)
DATASETS = ('HCP', 'MSC')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
)


def null_spatial_encoder(*pparams, **params):
    return None


def checkpoint_template(
    template: Mapping,
    update_weight: Optional[int] = 0,
    new_template: Optional[Mapping] = None,
):
    logging.info('Serialising template for checkpoint')
    template_concat = np.asarray(jnp.concatenate(
        (template['cortex_L'], template['cortex_R'])
    ))
    np.save(
        '/tmp/template.npy', template_concat, allow_pickle=False
    )
    if new_template is not None:
        new_template_concat = np.asarray(jnp.concatenate(
            (new_template['cortex_L'], new_template['cortex_R'])
        ))
        np.save(
            '/tmp/new_template.npy',
            new_template_concat,
            allow_pickle=False,
        )
    with open('/tmp/templateupdateweight.txt', 'w') as f:
        f.write(str(update_weight))


def main(last_checkpoint: Optional[int] = None):
    data_entities = []
    if 'MSC' in DATASETS:
        data_entities = data_entities + [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS, MSC_TASKS
            )
        ]
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_template.txt', 'r') as f:
            hcp_subjects = f.read().splitlines()
        data_entities = data_entities + [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects, HCP_TASKS
            )
        ]
    num_entities = len(data_entities)
    coor_L, coor_R = get_coors()
    model, _ = init_encoder_model(coor_L=coor_L)
    encode = eqx.filter_jit(model)
    encode = model
    epoch_history = []
    new_template = None
    update_weight = 0
    if last_checkpoint is not None:
        template_concat = np.load('/tmp/template.npy')
        # TODO: This should be a method of `encoder.temporal`
        template = {
            c: jax.lax.dynamic_slice(
                template_concat,
                (s[0],) + (0,) * (template_concat.ndim - 1),
                (s[1],) + template_concat.shape[1:],
            )
            for c, s in model.temporal.encoders[0].limits.items()
            if c in model.temporal.compartments
        }
        if (last_checkpoint % CHECKPOINT_INTERVAL == 0):
            new_template = None
            update_weight = 0
        else:
            new_template_concat = np.load('/tmp/new_template.npy')
            new_template = {
                c: jax.lax.dynamic_slice(
                    new_template_concat,
                    (s[0],) + (0,) * (new_template_concat.ndim - 1),
                    (s[1],) + new_template_concat.shape[1:],
                )
                for c, s in model.temporal.encoders[0].limits.items()
                if c in model.temporal.compartments
            }
            with open('/tmp/templateupdateweight.txt', 'r') as f:
                update_weight = int(f.read())
        start_epoch = last_checkpoint
    else:
        template = {'cortex_L': None, 'cortex_R': None}
        start_epoch = 0
    for i in range(start_epoch, MAX_EPOCH):
        entity = data_entities[i % num_entities]
        try:
            ds = entity.get('ds')
            if ds == 'MSC':
                subject = entity.get('subject')
                session = entity.get('session')
                task = entity.get('task')
                T = _get_data(
                    *get_msc_dataset(subject, session, task, get_confounds=True,),
                    denoising='mgtr+18',
                    filter_data=False,
                )
            elif ds == 'HCP':
                subject = entity.get('subject')
                session = entity.get('run')
                task = entity.get('task')
                T = _get_data(
                    *get_hcp_dataset(subject, session, task, get_confounds=True,),
                    denoising='mgtr+18',
                    filter_data=False,
                )
        except FileNotFoundError:
            logging.warning(
                f'Data entity {entity} is absent. '
                'Skipping'
            )
            continue
        logging.info(f'Epoch {i} (ds-{ds} sub-{subject} ses-{session} task-{task})')
        if jnp.any(jnp.isnan(T)):
            logging.warning(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        logging.info('Encoding')
        _, (template, new_new_template, new_update_weight, _, _) = encode(
            T=T,
            coor_L=coor_L,
            coor_R=coor_R,
            M=template,
            new_M=new_template,
            update_weight=update_weight,
        )
        if (
            jnp.any(jnp.isnan(new_new_template['cortex_L'])) or
            jnp.any(jnp.isnan(new_new_template['cortex_R']))
        ):
            logging.warning(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        new_template = new_new_template
        update_weight = new_update_weight
        if (
            ((i % num_entities == 0)
            and (new_template is not None))
        ):
            if template['cortex_L'] is not None:
                delta_norm_L = jnp.linalg.norm(
                    new_template['cortex_L'] - template['cortex_L']
                )
                delta_norm_R = jnp.linalg.norm(
                    new_template['cortex_R'] - template['cortex_R']
                )
                delta_norm = delta_norm_L + delta_norm_R
                epoch_history += [(delta_norm,)]
                logging.info(f'Epoch {i} delta norm: {delta_norm}')
            template = {
                compartment: model.temporal.rescale(new_template[compartment])
                for compartment in ('cortex_L', 'cortex_R')
            }
            update_weight = 0
            checkpoint_template(
                template=template,
                new_template=None,
            )
        elif (
            (i % CHECKPOINT_INTERVAL == 0) and
            (template['cortex_L'] is not None)
        ):
            checkpoint_template(
                template=template,
                new_template=new_template,
            )
    import matplotlib.pyplot as plt
    plt.plot([e[0] for e in epoch_history]) # training loss
    plt.savefig('/tmp/losses.png')
    with open('/tmp/template_deltas.txt', 'w') as f:
        f.write('\n'.join(str(e[0]) for e in epoch_history))
    assert 0


if __name__ == '__main__':
    main()
