# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
BrainPedia resampling
~~~~~~~~~~~~~~~~~~~~~
Resample the BrainPedia images to fsaverage5 and save them as GIFTI files.
"""
import pathlib
import lytemaps as nmaps
from nilearn.datasets import fetch_neurovault_ids


OUTPUT_DIR = '/tmp/brainpedia_resampled'


def main():
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    brainpedia = fetch_neurovault_ids(collection_ids=[1952])

    for image in brainpedia['images_meta']:
        image_path = image['absolute_path']
        #parent = pathlib.Path(image).parent
        base = pathlib.Path(image_path).with_suffix('').with_suffix('').name
        path_L = pathlib.Path(OUTPUT_DIR) / (base + ('_L.gii'))
        path_R = pathlib.Path(OUTPUT_DIR) / (base + ('_R.gii'))
        ds = image['study']
        ident = image['id']
        task = ''.join(
            e.capitalize() for e in
            image['task'].lower().split('_')
        )
        task = task[0].lower() + task[1:]
        dtype = ''.join(image['map_type'].lower().split())
        path_L = pathlib.Path(OUTPUT_DIR) / (
            f'id-{ident}_ds-{ds}_task-{task}_hemi-L_{dtype}.gii'
        )
        path_R = pathlib.Path(OUTPUT_DIR) / (
            f'id-{ident}_ds-{ds}_task-{task}_hemi-R_{dtype}.gii'
        )
        left, right = nmaps.transforms.mni152_to_fslr(
            image_path,
            fslr_density='32k',
        )
        left.to_filename(path_L)
        right.to_filename(path_R)


if __name__ == '__main__':
    main()
