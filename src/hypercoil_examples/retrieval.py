# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
File retrieval
~~~~~~~~~~~~~~
Retrieve some files or data
"""
from typing import Literal
from pkg_resources import resource_filename
import numpy as np


def get_neighbourhood(
    tpl: Literal['fsLR'] = 'fsLR',
    hemi: Literal['L', 'R'] = 'L',
    range: int = 1,
) -> np.ndarray:
    """
    Get the neighbourhood matrix for a given template and hemisphere

    Parameters
    ----------
    tpl : {'fsLR'}
        The template to use
    hemi : {'L', 'R'}
        The hemisphere to use
    range : int
        The range of the neighbourhood

    Returns
    -------
    neighbourhood : array
        The neighbourhood matrix
    """
    return np.load(
        resource_filename(
            'hypercoil_examples',
            f'data/surf/tpl-{tpl}_hemi-{hemi}_range-{range}_adjacency.npy',
        )
    )
