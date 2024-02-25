# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Thresholding geometry
~~~~~~~~~~~~~~~~~~~~~
Experiments to demonstrate the geometry of thresholding operations. Note that
thresholding will remove selectivity data from the hypersphere and restrict it
to the non-negative orthant of the unit hypercube.
"""
from itertools import permutations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hypercoil.engine import Tensor

KEY = jax.random.PRNGKey(11)


def random_maximal_clique(
    vertices: Tensor,
    direct_neighbours: Tensor,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    n_vertices = vertices.shape[0]
    vidx = jnp.arange(n_vertices)
    cliques = jnp.eye(n_vertices)
    for i in range(n_vertices):
        candidates = (cliques @ direct_neighbours)
        # We randomly reorder each row so that argmax doesn't
        # deterministically select the first instance
        rollidx = jax.random.choice(
            jax.random.fold_in(key, i),
            shape=(n_vertices,),
            a=n_vertices,
        )
        rolled = jax.vmap(jnp.roll, in_axes=(0, 0))(
            candidates * (1 - cliques), rollidx
        )
        select_idx = (
            (rolled.argmax(-1) - rollidx) % n_vertices
        )[..., None]
        jax.vmap(lambda x, i: x[i], in_axes=(0, 0))(
            candidates,
            select_idx,
        )
        cliques = jnp.where(
            candidates.max(-1, keepdims=True) == i,
            jnp.where(
                vidx == select_idx,
                1,
                cliques,
            ),
            cliques
        )
    return cliques


def main():
    # Experiment 1: Demonstrate that the top-k indices are the same as the
    # winner-takes all projection onto the vertices of the unit hypercube
    # intersected with the hyperplane 1^T x = r for number of surviving
    # vertices r
    n = 9
    r = 5
    seq = (0,) * (n - r) + (1,) * r
    vertices = np.unique(np.asarray(list(permutations(seq))), axis=0)
    data = jax.random.normal(shape=(100, n), key=jax.random.PRNGKey(0))
    data = data / jnp.linalg.norm(data, axis=-1, keepdims=True)
    proj = vertices[(data @ vertices.T).argmax(-1)]
    topk = (
        jnp.arange(n)[None, None, :] == jax.lax.top_k(data, r)[1][..., None]
    ).sum(-2)
    assert jnp.all(proj == topk)

    # Experiment 2: Demonstrate that the number of direct neighbours of each
    # vertex in the unit hypercube is equal to the number of surviving vertices
    # r times the number of removed vertices n - r
    direct_neighbours = (vertices @ vertices.T) == (r - 1)
    num_direct_neighbours = (direct_neighbours).sum(-1)
    assert jnp.all(num_direct_neighbours == (r * (n - r)))

    # Experiment 3: Demonstrate the existence of two sizes of maximal cliques,
    # which are formed starting from a vertex and then respectively permuting
    # a 1 or 0 in that vertex's coordinate.
    # - If a 0 is permuted, then the maximal clique is of size r + 1
    # - If a 1 is permuted, then the maximal clique is of size n - r + 1
    # - Trivially, each vertex participates in n - r cliques of the first type
    #   and r cliques of the second type
    # - If r = n / 2, then cliques of the two types are homotopic. In fact,
    #   cliques of one type for some vertex are the same as cliques of the other
    #   type for some other vertex
    # - If some vertex w participates in a clique of some type with vertex v,
    #   then w can participate in no other clique of the same type with the same
    #   vertex v
    cliques = random_maximal_clique(vertices, direct_neighbours, KEY)
    clique_size = cliques.sum(-1)
    assert jnp.all(
        jnp.logical_or(
            clique_size == r + 1,
            clique_size == n - r + 1,
        )
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(cliques, cmap='bone')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('/tmp/cliques.png')


if __name__ == '__main__':
    main()
