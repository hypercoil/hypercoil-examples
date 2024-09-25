# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Task and architectonic correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluate correspondences between parcel boundaries on one hand and task-evoked
activation patterns and architectonic areas on the other hand.
Here we're only considering group-average parcellations because we don't yet
have individualised task activation maps and will probably never have
individualised architectonic maps.
"""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from hypercoil.loss.functional import js_divergence
from hyve import (
    Cell,
    plotdef,
    add_surface_overlay,
    draw_surface_boundary,
    parcellate_colormap,
    plot_to_image,
    save_figure,
    select_active_parcels,
    surf_from_archive,
    surf_scalars_from_array,
    #surf_scalars_from_cifti,
    vertex_to_face,
)
from hypercoil_examples.atlas.energy import medial_wall_arrays, curv_arrays

GROUP_ATLAS_MATRICES = '/Users/rastkociric/Downloads/atlas-matrices/'
TASKS_MAPS = '/Users/rastkociric/Downloads/brainpedia_resampled/'
BRODMANN_AREAS = '/Users/rastkociric/Downloads/null_lL_WG33/Human.Brodmann09.32k_fs_LR.dlabel.nii'
VDG_AREAS = '/Users/rastkociric/Downloads/null_lL_WG33/Human.Composite_VDG11.32k_fs_LR.dlabel.nii'
MEDIAL_WALL = '/Users/rastkociric/Downloads/null_lL_WG33/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'


def load_atlases():
    return {
        e.parts[-1].split('_')[0].split('-')[1]: np.load(e)
        for e in Path(GROUP_ATLAS_MATRICES).glob('*.npy')
    }


def dice(a, b):
    # https://www.biorxiv.org/content/10.1101/306977v1
    inner = a @ b.swapaxes(-1, -2)
    numerator = 2 * inner
    coef = inner / (a @ np.sign(b).swapaxes(-1, -2))
    coef = np.where(np.isnan(coef), 1., coef)
    return numerator / (coef * a.sum(-1)[..., None] + b.sum(-1)[..., None, :])


def jaccard(a, b):
    # Note: This isn't vectorised the way that Dice is, but we don't need it
    # to be
    intersection = (a * b).sum(-1)
    union = (a + b).sum(-1) - intersection
    return intersection / union


def consolidate_step(a, b):
    assignment = (b @ a.T).argmax(0)
    a_consolidated = np.eye(b.shape[0])[assignment].T @ a
    return (
        a_consolidated[a_consolidated.sum(-1) != 0],
        (assignment == np.arange(len(assignment))).all(),
        (a_consolidated.sum(-1) != 0)
    )


def consolidate_labels(
    ref,
    atlas,
    max_steps: int = 100,
    single_step: bool = False,
):
    # This feels like cheating, and it probably is. We'll come up with a
    # better solution when we're not under time pressure.
    a = ref
    b = atlas
    a_stop = False
    b_stop = False
    step = 0
    while (not a_stop and not b_stop) and step < max_steps:
        logging.info(f'Consolidation step {step}')
        b, b_stop, mask = consolidate_step(b, a)
        if single_step:
            a = a[mask]
            break
        a, a_stop, mask = consolidate_step(a, b)
        step += 1
    logging.info(f'Consolidated into {a.shape[0]} labels after {step} steps')
    return a, b


def arch_plot():
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | Cell() | layout << (1 / 3)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(view='dorsal'),
        3: dict(
            hemisphere='right',
            view='lateral',
        ),
        4: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            plot=False,
            allow_multihemisphere=False,
        ),
        add_surface_overlay(
            'medialwall',
            surf_scalars_from_array('medialwall', is_masked=False),
            vertex_to_face('medialwall', interpolation='mode'),
        ),
        add_surface_overlay(
            'reference',
            surf_scalars_from_array('reference', allow_multihemisphere=False,),
            parcellate_colormap('reference'),
            #vertex_to_face('reference', interpolation='mode'),# points_suffix='_points'),
        ),
        add_surface_overlay(
            'parcellation',
            #parcellate_colormap('parcellation'),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                #target_domain='vertex',
                copy_values_to_boundary=True,
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        add_surface_overlay(
            'reference_boundary',
            parcellate_colormap('reference'),
            draw_surface_boundary(
                'reference',
                'reference_boundary',
                #target_domain='vertex',
                copy_values_to_boundary=True,
                target_domain='face',
                num_steps=1,
                v2f_interpolation='mode',
            ),
        ),
        vertex_to_face('reference', interpolation='mode'),
        # add_surface_overlay(
        #     'curv',
        #     surf_scalars_from_array('curv', is_masked=False),
        #     vertex_to_face('curv', interpolation='mean'),
        # ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 500),
            #canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    return plot_f


def task_plot():
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * layout * layout
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(view='anterior'),
        3: dict(view='dorsal'),
        4: dict(view='ventral'),
        5: dict(view='posterior'),
        6: dict(
            hemisphere='right',
            view='medial',
        ),
        7: dict(
            hemisphere='right',
            view='lateral',
        ),
        8: dict(
            elements=['scalar_bar'],
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'medialwall',
            surf_scalars_from_array('medialwall', is_masked=False),
            vertex_to_face('medialwall', interpolation='mode'),
        ),
        add_surface_overlay(
            'activation',
            surf_scalars_from_array('activation', plot=True),
        ),
        add_surface_overlay(
            'parcellation',
            surf_scalars_from_array('parcellation', plot=False),
            select_active_parcels(
                'parcellation',
                'activation',
                parcel_coverage_threshold=0.5,
                use_abs=True,
            ),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                copy_values_to_boundary=True,
                target_domain='face',
                num_steps=1,
                v2f_interpolation='mode',
            ),
        ),
        vertex_to_face('activation', interpolation='mean'),
        add_surface_overlay(
            'curv',
            surf_scalars_from_array('curv', is_masked=False),
            vertex_to_face('curv', interpolation='mean'),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(3200, 400),
            #canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    return plot_f


def architectonic_correspondence(atlases, consolidate=True):
    medial_wall = nb.load(MEDIAL_WALL).get_fdata().astype(bool)
    brodmann = nb.load(BRODMANN_AREAS).get_fdata()[~medial_wall].astype(int)
    vdg = nb.load(VDG_AREAS).get_fdata()[~medial_wall].astype(int)
    plot_f = arch_plot()
    medialwall_L, medialwall_R = medial_wall_arrays()
    curv_L, curv_R = curv_arrays()
    for name, ref in (('brodmann', brodmann), ('vdg', vdg)):
        dice_coef = {}
        jaccard_coef = {}
        jsd_coef = {}
        ref = np.eye(ref.max() + 1)[ref].T
        if name == 'brodmann':
            ref = ref[2:] # 0: ???, 1: medial wall
        ref = ref[ref.sum(-1) != 0]
        for atlas_name, atlas in atlases.items():
            # assignment = (ref @ atlas.T).argmax(0)
            # consolidated = np.eye(ref.shape[0])[assignment].T @ atlas
            # consolidated = consolidated[consolidated.sum(-1) != 0]
            if consolidate:
                ref_c, atlas_c = consolidate_labels(ref, atlas, single_step=True)
            else:
                raise NotImplementedError
            # Discretise the data
            ref_c = (
                (ref_c == ref_c.max(-2, keepdims=True)) * (ref_c != 0)
            ).astype(float) # fp required for JS divergence
            dice_coef[atlas_name] = 1 - dice(
                ref_c[..., None, :], atlas_c[..., None, :]
            ).squeeze().mean()
            jaccard_coef[atlas_name] = 1 - jaccard(
                ref_c, atlas_c
            ).squeeze().mean()
            jsd = js_divergence(ref_c, atlas_c, axis=0)
            jsd = np.where(jsd < 0, 0, jsd) # precision errors
            jsd_coef[atlas_name] = (np.sqrt(jsd)).mean()
        # Plot the results
        plot_f(
            template='fsLR',
            surf_projection='veryinflated',
            surf_color=(0.8, 0.8, 0.8),
            window_size=(600, 500),
            hemisphere=['left', 'right', 'both'],
            views={
                'left': ('medial', 'lateral'),
                'right': ('medial', 'lateral'),
                'both': ('dorsal',),
            },
            #theme=pv.themes.DarkTheme(),
            output_dir='/tmp',
            fname_spec=f'scalars-{name}',
            load_mask=True,
            reference_array_left=ref[..., :(~medialwall_L).sum()].argmax(0),
            reference_array_right=ref[..., (~medialwall_L).sum():].argmax(0),
            reference_cmap='modal',
            reference_alpha=0.5,
            reference_boundary_cmap='modal',
            parcellation_array_left=atlases['groupTemplate'][..., :(~medialwall_L).sum()].argmax(0) + 1,
            parcellation_array_right=atlases['groupTemplate'][..., (~medialwall_L).sum():].argmax(0) + 1,
            parcellation_color='#444444',
            #parcellation_cmap='network',
            parcellation_alpha=0.9,
            # curv_array_left=curv_L,
            # curv_array_right=curv_R,
            # curv_cmap='gray',
            # curv_clim=(-5e-1, 5e-1),
            # curv_alpha=0.3,
            medialwall_array_left=medialwall_L,
            medialwall_array_right=medialwall_R,
            medialwall_cmap='binary',
            medialwall_clim=(0.99, 1),
            medialwall_below_color=(0, 0, 0, 0),
            surf_style={'lighting': False},
            parallel_projection=True,
        )
        for measure, result in (
            ('Dice', dice_coef),
            ('Jaccard', jaccard_coef),
            ('JS distance', jsd_coef),
        ):
            plt.figure(figsize=(6, 4), layout='tight')
            df = pd.DataFrame(result, index=[None]).melt().rename(
                columns={'variable': 'model', 'value': 'score'}
            )
            ax = sns.barplot(df, x='score', y='model', color='grey')
            ax.bar_label(ax.containers[0], color='black', fontweight='bold', fontsize=6)
            sns.despine()
            plt.title(measure)
            plt.yticks(rotation=45)
            ax.yaxis.set_ticks_position('none')
            plt.savefig('/tmp/{}_{}.svg'.format(name, measure.replace(' ', '')), dpi=300)
    plt.close('all')


def prep_maps():
    activation_map_paths = Path(TASKS_MAPS).glob('*')
    maps_paths_parsed = tuple((e.parts[-1], e) for e in activation_map_paths)
    maps_paths_parsed = tuple(
        (e[0].split('_')[0], e[0].split('_')[-2], e[1])
        for e in maps_paths_parsed
    )
    maps_paths_df = pd.DataFrame(
        data=maps_paths_parsed,
        columns=['id', 'hemi', 'path'],
    )
    maps_paths_df = maps_paths_df.pivot(
        columns='hemi',
        index='id',
        values='path',
    )
    medialwall_L, medialwall_R = medial_wall_arrays()
    hemisphere_boundary = (~medialwall_L).sum()
    masks = {
        'hemi-L': medialwall_L,
        'hemi-R': medialwall_R,
    }
    return maps_paths_df, masks, hemisphere_boundary


def task_reconstruction(atlases):
    maps_paths_df, masks, hemisphere_boundary = prep_maps()
    for hemisphere in ('hemi-L', 'hemi-R'):
        logging.info(f'Processing hemisphere {hemisphere}')
        paths = maps_paths_df.reset_index()[hemisphere].dropna()
        logging.info('Loading activation maps --- this may take a while')
        activation_maps = np.stack([
            nb.load(str(e)).darrays[0].data[~masks[hemisphere]]
            for e in paths
        ])
        logging.info('Establishing reconstruction ceiling')
        max_rank = max([v.shape[0] for v in atlases.values()]) // 2
        S = np.linalg.svd(
            activation_maps,
            full_matrices=False,
            compute_uv=False,
        ) ** 2
        ceiling = S.cumsum() / S.sum()
        ceiling = np.concatenate(([0.], ceiling))
        plt.figure(figsize=(6, 4), layout='tight')
        varexp = {}
        varexp_norm = {}
        for atlas_name, atlas in atlases.items():
            logging.info(f'Processing atlas {atlas_name}')
            match hemisphere:
                case 'hemi-L':
                    atlas = atlas[..., :hemisphere_boundary]
                case 'hemi-R':
                    atlas = atlas[..., hemisphere_boundary:]
            atlas = atlas[atlas.sum(-1) != 0]
            logging.info('Reconstructing activation maps')
            theta = np.linalg.lstsq(atlas.T, activation_maps.T)[0]
            residuals = (activation_maps - (theta.T @ atlas))
            varexp[atlas_name] = 1 - (residuals ** 2).sum() / (activation_maps ** 2).sum()
            num_parcels = atlas.shape[0]
            varexp_norm[atlas_name] = varexp[atlas_name] / ceiling[num_parcels]
            plt.scatter(num_parcels, varexp[atlas_name])

        plt.plot(ceiling[:int(max_rank * 1.25)], color='grey')
        sns.despine()
        plt.xlabel('Atlas dimension')
        plt.ylabel('% variance explained')
        plt.legend(list(atlases.keys()) + ['ceiling'])
        plt.title(
            'Reconstruction quality: {hemi} hemisphere'.format(hemi=(
                'left' if hemisphere == 'hemi-L' else 'right'
            ))
        )
        plt.savefig(f'/tmp/reconstruction_{hemisphere}.svg', dpi=300)
        plt.figure(figsize=(6, 4), layout='tight')
        df = pd.DataFrame(varexp_norm, index=[None]).melt().rename(
            columns={'variable': 'model', 'value': 'score'}
        )
        ax = sns.barplot(df, x='score', y='model', color='grey')
        ax.bar_label(ax.containers[0], color='black', fontweight='bold', fontsize=6)
        sns.despine()
        plt.title('Variance explained (normalised to ceiling)')
        plt.yticks(rotation=45)
        ax.yaxis.set_ticks_position('none')
        plt.savefig('/tmp/reconstructionnorm_{hemisphere}.svg', dpi=300)
    plt.close('all')


def task_visualisation(atlases):
    maps_paths_df, masks, hemisphere_boundary = prep_maps()
    medialwall_L, medialwall_R = medial_wall_arrays()
    curv_L, curv_R = curv_arrays()
    plot_f = task_plot()
    for idx in (
        37, 84, 144, 263, 371, 500, 733,
        1000, 1233, 1407, 1671, 1832, 2000,
        2333, 2500, 2671, 2832, 3000,
        3333, 3500, 3671, 3832, 4000,
    ):
        logging.info(f'Plotting activation maps for task {idx}')
        activation_left = nb.load(
            maps_paths_df.iloc[idx]['hemi-L']
        ).darrays[0].data[~masks['hemi-L']]
        activation_right = nb.load(
            maps_paths_df.iloc[idx]['hemi-R']
        ).darrays[0].data[~masks['hemi-R']]
        activation_left = np.where(
            np.abs(activation_left) < 3.,
            0.,
            activation_left,
        )
        activation_right = np.where(
            np.abs(activation_right) < 3.,
            0.,
            activation_right,
        )
        plot_f(
            template="fsLR",
            surf_projection='veryinflated',
            surf_color=(0.3, 0.3, 0.3),
            load_mask=True,
            parcellation_array=atlases['groupTemplate'].argmax(0) + 1,
            parcellation_color='#77ff77',
            #parcellation_color='black',
            #parcellation_color='white',
            parcellation_alpha=0.8,
            activation_array_left=activation_left,
            activation_array_right=activation_right,
            activation_cmap='magma',
            activation_cmap_negative='bone',
            activation_below_color=(0, 0, 0, 0),
            medialwall_array_left=medialwall_L,
            medialwall_array_right=medialwall_R,
            medialwall_cmap='binary',
            medialwall_clim=(0.99, 1),
            medialwall_below_color=(0, 0, 0, 0),
            curv_array_left=curv_L,
            curv_array_right=curv_R,
            curv_cmap='gray',
            curv_clim=(-5e-1, 5e-1),
            curv_alpha=0.1,
            # surf_color=(0.3, 0.3, 0.3),
            # surf_style={'lighting': False},
            # parallel_projection=True,
            hemisphere=['left', 'right', 'both'],
            views={
                'left': ('medial', 'lateral'),
                'right': ('medial', 'lateral'),
                'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
            },
            output_dir='/tmp',
            fname_spec=f'scalars-taskactivation{idx}',
            window_size=(600, 400),
            #theme=pv.themes.DarkTheme(),
        )
    assert 0


def main():
    atlases = load_atlases()
    task_visualisation(atlases)
    task_reconstruction(atlases)
    architectonic_correspondence(atlases)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] - %(message)s',
    )
    main()
