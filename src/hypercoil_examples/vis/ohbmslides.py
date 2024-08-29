# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``hyve`` examples
~~~~~~~~~~~~~~~~~
Examples of how to use the ``hyve`` package. These examples are used in the
OHBM 2024 presentation.
"""
import pathlib, pickle
import lytemaps as nmaps
import nibabel as nb
import numpy as np
import pyvista as pv
import templateflow.api as tflow
from hyve import (
    Cell,
    ColGroupSpec,
    plotdef,
    add_edge_variable,
    add_network_data,
    add_node_variable,
    add_surface_overlay,
    build_network,
    draw_surface_boundary,
    node_coor_from_regions,
    surf_from_archive,
    surf_scalars_from_array,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    parcellate_colormap,
    planar_sweep_camera,
    plot_to_image,
    points_scalars_from_nifti,
    pyplot_element,
    save_figure,
    save_snapshots,
    scatter_into_parcels,
    svg_element,
    text_element,
    vertex_to_face,
)
from hyve_examples import get_schaefer400_cifti


def animate_gif(fnames, out, duration=500, loop=0):
    #TODO: Implement animation as a primitive in hyve. It's incredibly
    # straightforward if we're willing to add a dependency on imageio.
    import imageio.v3 as iio
    images = [iio.imread(fname) for fname in fnames]
    iio.imwrite(out, images, duration=duration, loop=loop)

def turntable_surf():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_gifti('curv', is_masked=False),
        plot_to_image(),
        planar_sweep_camera(n_angles=36),
        save_snapshots(),
    )
    curv = nmaps.datasets.fetch_fsaverage()['sulc']
    plot_f(
        template='fsaverage',
        surf_projection='inflated',
        surf_style={
            'pbr': True,
            'metallic': 0.05,
            'roughness': 0.1,
            'specular': 0.5,
            'specular_power': 15,
            # 'diffuse': 1,
        },
        hemisphere='left',
        curv_gifti_left=str(curv.L),
        curv_gifti_right=str(curv.R),
        surf_scalars_cmap='bone',
        surf_scalars_clim=(-1e-1, 1e-1),
        output_dir='/tmp',
        fname_spec=(
            'scalars-{surfscalars}_view-{view}_index-{index:03d}_'
            'hemisphere-{hemisphere}_scene.png'),
        window_size=(600, 500),
        empty_builders=True,
    )
    fnames = sorted(pathlib.Path('/tmp').glob('scalars-curv*.png'))
    animate_gif(fnames, out='/tmp/turntable_surf.gif', duration=100, loop=0)


def turntable_pointcloud():
    plot_f = plotdef(
        surf_from_archive(),
        points_scalars_from_nifti('wm'),
        plot_to_image(),
        planar_sweep_camera(n_angles=36),
        save_snapshots(),
    )
    wm = tflow.get(
        'MNI152NLin2009cAsym', resolution=2, suffix='probseg', label='WM'
    )
    plot_f(
        template='fsaverage',
        surf_projection='pial',
        surf_alpha=0.1,
        hemisphere='both',
        wm_nifti=str(wm),
        points_scalars_cmap='inferno',
        points_scalars_clim=(0.8, 1),
        output_dir='/tmp',
        fname_spec=(
            'scalars-{pointsscalars}_view-{view}_'
            'index-{index:03d}_scene.png'
        ),
        window_size=(600, 500),
        empty_builders=True,
    )
    fnames = sorted(pathlib.Path('/tmp').glob('scalars-wm*left*.png'))
    animate_gif(fnames, out='/tmp/turntable_pointcloud.gif', duration=100, loop=0)


def turntable_network():
    plot_f = plotdef(
        surf_from_archive(),
        add_network_data(
            add_node_variable('tile'),
            add_edge_variable('conn'),
        ),
        build_network('tile'),
        plot_to_image(),
        planar_sweep_camera(n_angles=36),
        save_snapshots(),
    )
    node_coor = np.stack(
        np.meshgrid(
            # np.asarray((0, 20, 40)),
            # np.asarray((-40, -20, 0, 20, 40)),
            np.asarray((1, 40)),
            np.asarray((-60, -20, 20)),
            np.asarray((-20, 20, 60)),
        )
    ).reshape(3, -1)
    plot_f(
        template='fsaverage',
        hemisphere='right',
        surf_projection='pial',
        surf_alpha=0.2,
        node_radius=5,
        node_color='index',
        node_style={
            'pbr': True, 'metallic': 0.3, 'roughness': 0.1,
            'specular': 0.5, 'specular_power': 15,
        },
        edge_color='src',
        edge_clim=(0, node_coor.shape[1]),
        edge_radius=2,
        edge_style={
            'pbr': True, 'metallic': 0.3, 'roughness': 0.1,
            'specular': 0.5, 'specular_power': 15,
        },
        node_coor=node_coor.T,
        tile_nodal=np.ones(node_coor.shape[1]),
        conn_adjacency=np.ones((node_coor.shape[1], node_coor.shape[1])),
        output_dir='/tmp',
        fname_spec=(
            'net-tile_view-{view}_'
            'index-{index:03d}_scene.png'
        ),
        window_size=(600, 500),
        empty_builders=True,
    )
    fnames = sorted(pathlib.Path('/tmp').glob('net-tile*.png'))
    animate_gif(fnames, out='/tmp/turntable_network.gif', duration=100, loop=0)


def layout_5view():
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
    return layout


def example_parcel_arrays():
    array_left = np.load('/tmp/cortexL.npy')
    array_right = np.load('/tmp/cortexR.npy')
    return array_left, array_right


def example_annular_array():
    with open('/tmp/concatannular.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['projection']


def medial_wall_arrays():
    mwl = nb.load(str(
        tflow.get('fsLR', density='32k', hemi='L', desc='nomedialwall')
    )).darrays[0].data
    mwr = nb.load(str(
        tflow.get('fsLR', density='32k', hemi='R', desc='nomedialwall')
    )).darrays[0].data
    return mwl, mwr


def blank_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsaverage',
        # parcel_apply_mask=True,
        # parcel_is_masked=False,
        surf_projection='pial',
        surf_color='#AABBFF',
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
    )


def snaps_5view():
    plot_f = plotdef(
        surf_from_archive(),
        plot_to_image(),
        save_snapshots(),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec='view-5fig_desc-0snaps_angle-{view}_hemi-{hemisphere}',
        load_mask=True,
    )


def fig_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'view-5fig_desc-1fig',
        load_mask=True,
    )


def parcels_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            is_masked=True,
            allow_multihemisphere=False,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    array_left, array_right = example_parcel_arrays()
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcellation_array_left=array_left.argmax(0) + 1,
        parcellation_array_right=array_right.argmax(0) + 1,
        surf_scalars_cmap='tab20b',
        surf_scalars_clim=(1, 400),
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'view-5fig_desc-2parcels',
        load_mask=True,
    )


def cmap_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            is_masked=True,
            allow_multihemisphere=False,
        ),
        parcellate_colormap('parcellation'),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    array_left, array_right = example_parcel_arrays()
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcellation_array_left=array_left.argmax(0) + 1,
        parcellation_array_right=array_right.argmax(0) + 1,
        surf_scalars_cmap='network',
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'view-5fig_desc-3cmap',
        load_mask=True,
    )


def boundary_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            is_masked=True,
            allow_multihemisphere=False,
        ),
        parcellate_colormap('parcellation'),
        draw_surface_boundary(
            'parcellation',
            'parcellation',
            copy_values_to_boundary=True,
            target_domain='face',
            num_steps=0,
            v2f_interpolation='mode',
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    array_left, array_right = example_parcel_arrays()
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcellation_array_left=array_left.argmax(0) + 1,
        parcellation_array_right=array_right.argmax(0) + 1,
        surf_scalars_cmap='network',
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'view-5fig_desc-4boundary',
        load_mask=True,
    )


def final_5view():
    layout = layout_5view()
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'curv',
            surf_scalars_from_gifti('curv', is_masked=False),
            vertex_to_face('curv', interpolation='mode'),
        ),
        add_surface_overlay(
            'parcellation',
            surf_scalars_from_array(
                'parcellation',
                is_masked=True,
                allow_multihemisphere=False,
            ),
            parcellate_colormap('parcellation'),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                copy_values_to_boundary=True,
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 600),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    array_left, array_right = example_parcel_arrays()
    curv = nmaps.datasets.fetch_fslr()['sulc']
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcellation_array_left=array_left.argmax(0) + 1,
        parcellation_array_right=array_right.argmax(0) + 1,
        parcellation_cmap='network',
        curv_gifti_left=str(curv.L),
        curv_gifti_right=str(curv.R),
        curv_cmap='bone',
        curv_clim=(-1e-1, 1e-1),
        curv_alpha=0.3,
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'view-5fig_desc-5overlays',
        load_mask=True,
    )


def annular_plot(data, figsize):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.real(data),
        np.imag(data),
        s=0.01,
        c=np.angle(data),
        cmap='twilight',
    )
    fig.tight_layout()
    return fig


def annular_vis():
    array = example_annular_array()[0]
    array_mag = np.abs(array)
    array_phase = np.angle(array)

    side = Cell() | Cell() << (1 / 2)
    centre = Cell() | Cell() | Cell() << (1 / 3)
    layout = side / centre / side << (1 / 3)
    layout = Cell() / layout << (1 / 30)
    layout = layout / Cell() << (15 / 16)
    layout = layout.annotate({
        0: dict(elements=['title']),
        1: dict(
            hemisphere='left',
            view='lateral',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(elements=['projectcartoon']),
        4: dict(view='dorsal'),
        5: dict(elements=['projectpyplot']),
        6: dict(
            hemisphere='left',
            view='medial',
        ),
        7: dict(
            hemisphere='right',
            view='medial',
        ),
        8: dict(elements=[{'scalar_bar': ('phase')}]),
    })
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array('magnitude', is_masked=True, plot=False),
        add_surface_overlay(
            'medialwall',
            surf_scalars_from_array('medialwall', is_masked=False),
        ),
        add_surface_overlay(
            'phase',
            surf_scalars_from_array(
                'phase',
                is_masked=True,
            ),
        ),
        add_surface_overlay(
            'curv',
            surf_scalars_from_gifti('curv', is_masked=False),
        ),
        plot_to_image(),
        text_element(
            name='title',
            bounding_box_height=128,
            font_size_multiplier=1.0,
            font_color='#cccccc',
            font_outline_multiplier=0,
        ),
        pyplot_element(
            name='projectpyplot',
            plotter=annular_plot,
            data=array,
            figsize=(6, 5),
        ),
        svg_element(
            name='projectcartoon',
            src_file='/tmp/OHBMvefigInsert.svg',
            height=262,
            width=223,
        ),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 1000),
            canvas_color=(0, 0, 0),
            fname_spec='desc-blank',
            scalar_bar_action='collect',
        ),
    )
    medialwall_left, medialwall_right = medial_wall_arrays()
    curv = nmaps.datasets.fetch_fslr()['sulc']
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        surf_color='#AAFF66',
        phase_array=array_phase,
        magnitude_array=array_mag,
        phase_cmap='twilight',
        phase_clim=(-np.pi, np.pi),
        phase_clim_percentile=False,
        phase_alpha='magnitude',
        phase_amap=(0, 1),
        phase_alim=(0, 0.15),
        phase_alim_percentile=False,
        curv_gifti_left=str(curv.L),
        curv_gifti_right=str(curv.R),
        curv_cmap='gray',
        curv_clim=(-5e-1, 5e-1),
        curv_alpha=0.3,
        medialwall_array_left=1 - medialwall_left,
        medialwall_array_right=1 - medialwall_right,
        medialwall_cmap='binary',
        medialwall_clim=(0.99, 1),
        medialwall_below_color=(0, 0, 0, 0),
        window_size=(600, 500),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'scalars-annular',
        load_mask=True,
        title_element_content='Annular decomposition: component 7',
        phase_scalar_bar_style={
            'name': 'phase',
            'orientation': 'h',
        },
    )


def ggseg_mimic():
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | layout << (1 / 2)
    annotations = {
        0: dict(
            hemisphere='right',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='lateral',
        ),
        2: dict(
            hemisphere='right',
            view='medial',
        ),
        3: dict(
            hemisphere='left',
            view='medial',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti(
            'parcellation',
            plot=False,
        ),
        add_surface_overlay(
            'data',
            scatter_into_parcels('data', 'parcellation'),
            vertex_to_face('data', interpolation='mode'),
        ),
        add_surface_overlay(
            'parcellation_boundary',
            draw_surface_boundary(
                'parcellation',
                'parcellation_boundary',
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(800, 600),
            scalar_bar_action='collect',
        ),
    )
    parcellation_cifti = get_schaefer400_cifti()
    plot_f(
        template='fsLR',
        surf_color='#999999',
        surf_projection='veryinflated',
        parcellation_cifti=parcellation_cifti,
        data_parcellated=np.arange(400) + 1,
        data_cmap='viridis',
        data_clim=(0, 400),
        data_clim_percentile=False,
        parcellation_boundary_color='black',
        window_size=(600, 500),
        hemisphere=['left', 'right'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
        },
        output_dir='/tmp',
        fname_spec=f'desc-ggseg',
        surf_style={'lighting': False},
        parallel_projection=True,
        empty_builders=True,
    )


def surfplot_reference():
    # Dan J Gale's surfplot example from:
    # https://surfplot.readthedocs.io/en/latest/auto_examples/plot_tutorial_05.html
    from surfplot import Plot
    from surfplot.datasets import load_example_data

    surfaces = nmaps.datasets.fetch_fslr()
    default = load_example_data(join=True)
    fronto = load_example_data('frontoparietal', join=True)
    lh, rh = surfaces['inflated']
    sulc_lh, sulc_rh = surfaces['sulc']

    p = Plot(lh, rh)
    p.add_layer(
        {'left': sulc_lh, 'right': sulc_rh},
        cmap='binary_r',
        cbar=False,
    )
    # default mode network associations
    p.add_layer(
        default,
        cmap='GnBu_r',
        cbar_label='Default mode',
        alpha=.5,
    )
    p.add_layer(
        fronto,
        cmap='YlOrBr_r',
        cbar_label='Frontoparietal',
        alpha=.5,
    )
    fig = p.build()
    fig.tight_layout()
    fig.savefig('/tmp/surfplot_reference.png', dpi=300)


def surfplot_mimic():
    from surfplot.datasets import load_example_data
    stack = Cell() / Cell() << (1 / 2)
    layout = stack | stack << (1 / 2)
    layout = layout / stack << (7 / 8)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(
            hemisphere='right',
            view='medial',
        ),
        4: dict(elements=[{'scalar_bar': ('Default mode',)}]),
        5: dict(elements=[{'scalar_bar': ('Frontoparietal',)}])
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'sulc',
            surf_scalars_from_gifti('sulc', is_masked=False),
            #vertex_to_face('sulc', interpolation='mean'),
        ),
        add_surface_overlay(
            'default',
            surf_scalars_from_array('default', is_masked=False),
            #vertex_to_face('default', interpolation='mean'),
        ),
        add_surface_overlay(
            'fronto',
            surf_scalars_from_array('fronto', is_masked=False),
            #vertex_to_face('fronto', interpolation='mean'),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(900, 800),
            scalar_bar_action='collect',
        ),
    )
    sulc = nmaps.datasets.fetch_fslr()['sulc']
    plot_f(
        template='fsLR',
        surf_projection='inflated',
        sulc_gifti_left=str(sulc.L),
        sulc_gifti_right=str(sulc.R),
        sulc_cmap='binary_r',
        sulc_alpha=0.7,
        default_array=load_example_data(join=True),
        default_cmap='GnBu_r',
        default_clim=(0, 100),
        default_clim_percentile=True,
        default_alpha=0.5,
        default_below_color=(0, 0, 0, 0),
        fronto_array=load_example_data('frontoparietal', join=True),
        fronto_cmap='YlOrBr_r',
        fronto_clim=(0, 100),
        fronto_clim_percentile=True,
        fronto_alpha=0.5,
        fronto_below_color=(0, 0, 0, 0),
        window_size=(600, 500),
        hemisphere=['left', 'right'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
        },
        output_dir='/tmp',
        fname_spec=f'desc-surfplot',
        default_scalar_bar_style={
            'name': 'Default mode',
            'orientation': 'h',
        },
        fronto_scalar_bar_style={
            'name': 'Frontoparietal',
            'orientation': 'h',
        },
    )


def glassbrain_reference():
    from nilearn.plotting import plot_glass_brain
    from nilearn import datasets

    stat_img = datasets.load_sample_motor_activation_image()
    plot_glass_brain(
        stat_img,
        threshold=3,
        display_mode="lyrz",
        black_bg=True,
        plot_abs=False,
        output_file='/tmp/ref-glassbrain.png',
    )
    breakpoint()


def glassbrain_mimic():
    from nilearn import datasets
    layout = Cell() | Cell() | Cell() | Cell() << (1 / 4)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='both',
            view='posterior',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(
            hemisphere='both',
            view='dorsal',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        points_scalars_from_nifti('stat'),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 300),
            canvas_color=(0, 0, 0),
            scalar_bar_action='collect',
        ),
    )
    stat_img = datasets.load_sample_motor_activation_image()
    plot_f(
        template='fsaverage',
        surf_projection='pial',
        surf_alpha=0.2,
        stat_nifti=stat_img,
        points_scalars_cmap='YlOrBr_r',
        points_scalars_clim=(3.0, 7.5),
        points_scalars_cmap_negative='Blues_r',
        points_scalars_clim_negative=(3.0, 7.5),
        points_scalars_clim_percentile=False,
        hemisphere=['left', 'right', None],
        views={
            'left': ('lateral',),
            'right': ('lateral',),
            'both': ('dorsal', 'posterior'),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec='desc-glassbrain',
        window_size=(600, 500),
        empty_builders=True,
    )


def brainnet_mimic():
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | layout << (1 / 2)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='both',
            view='posterior',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(
            hemisphere='both',
            view='dorsal',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        points_scalars_from_nifti('parcellation', plot=False),
        add_network_data(
            add_node_variable('data'),
            add_edge_variable(
                'conn',
                emit_degree=True,
                threshold=0.9,
                percent_threshold=True,
            ),
        ),
        node_coor_from_regions('parcellation'),
        build_network('data'),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(900, 750),
            scalar_bar_action='collect',
        ),
    )
    # TODO: node_coor_from_parcels should also be given a volumetric
    #       analogue. The below snippet basically outlines how to do it.
    nodal_img = nb.load('/tmp/motor_system_rois.nii.gz')
    nodal = nodal_img.get_fdata().astype(int)
    node_ids = sorted(set(np.unique(nodal)).difference({0}))
    adj = np.load('/tmp/adj_mat_1.npy')
    plot_f(
        template='fsaverage',
        surf_projection='pial',
        surf_alpha=0.2,
        surf_style={
            'pbr': True, 'metallic': 0.3,
            'specular': 0.5, 'specular_power': 15,
        },
        parcellation_nifti='/tmp/motor_system_rois.nii.gz',
        node_color='conn_degree',
        node_cmap='inferno',
        node_radius='conn_degree',
        node_rmap=(3, 10),
        node_style={
            'pbr': True, 'metallic': 0.3,
            'specular': 0.5, 'specular_power': 15,
        },
        #edge_color='black', #TODO -- important! This doesn't work!
        edge_color='conn_sgn',
        edge_cmap='gray',
        edge_radius='conn_val',
        edge_rmap=(1, 3),
        edge_style={
            'pbr': True, 'metallic': 0.3,
            'specular': 0.5, 'specular_power': 15,
        },
        #node_coor=node_coor.T,
        data_nodal=np.ones(len(node_ids)), # TODO: should be optional
        conn_adjacency=adj,
        hemisphere=['left', 'right', None],
        views={
            'left': ('lateral',),
            'right': ('lateral',),
            'both': ('dorsal', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec='desc-brainnet',
        window_size=(600, 500),
        empty_builders=True,
    )


def main():
    brainnet_mimic()


if __name__ == '__main__':
    main()
