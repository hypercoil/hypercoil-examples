# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
All parcels plot
~~~~~~~~~~~~~~~~
Visualise all parcels.
"""
import pyvista as pv
from hyve import (
    Cell,
    ColGroupSpec,
    plotdef,
    surf_from_archive,
    surf_scalars_from_array,
    text_element,
    plot_to_image,
    save_figure,
)


def visdef():
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | Cell() | layout << (1 / 3)
    layout = Cell() | layout << (1 / 35)
    layout = layout | Cell() << (35 / 36)
    annotations = {
        0: dict(elements=['subtitle']),
        1: dict(
            hemisphere='left',
            view='lateral',
        ),
        2: dict(
            hemisphere='left',
            view='medial',
        ),
        3: dict(view='dorsal'),
        4: dict(
            hemisphere='right',
            view='lateral',
        ),
        5: dict(
            hemisphere='right',
            view='medial',
        ),
        6: dict(elements=['scalar_bar']),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array('parcel'),
        plot_to_image(),
        text_element(
            name='subtitle',
            content='{surfscalars}',
            #angle=270,
            angle=90,
            bounding_box_height=128,
            font_size_multiplier=0.8,
            font_color='#cccccc',
            font_outline_multiplier=0,
        ),
        save_figure(
            layout_kernel=layout,
            group_spec = [
                ColGroupSpec(
                    variable='surfscalars',
                    max_levels=4,
                ),
            ],
            sort_by=['surfscalars'],
            padding=0,
            canvas_size=(1200, 1800),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-parcels_page-{page}',
            scalar_bar_action='collect',
        ),
    )
    return plot_f


def main():
    import jax.numpy as jnp
    array_left = jnp.load('/tmp/DualModelRun0Fail/UNetCortexL.npy')
    array_right = jnp.load('/tmp/DualModelRun0Fail/UNetCortexR.npy')
    plot_f = visdef()
    plot_f(
        template='fsLR',
        #load_mask=True,
        parcel_array_left=array_left,
        parcel_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='plasma',
        surf_scalars_clim=(0.8, 0.95),
        surf_scalars_clim_percentile=True,
        #surf_scalars_clim=(0.05, 0.15),
        #surf_scalars_clim_percentile=False,
        surf_scalars_below_color='#334466',
        window_size=(400, 300),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
    )
    assert 0


if __name__ == '__main__':
    main()
