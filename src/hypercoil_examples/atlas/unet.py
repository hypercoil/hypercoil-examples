# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ELLGAT u-net

A u-net model composed of ELLGAT blocks. We draw considerable inspiration from
SUGAR, a GAT-based u-net model for brain surface registration. The code for
this model is available at
https://github.com/IndiLab/SUGAR/blob/main/models/gatunet_model.py
"""
from collections import defaultdict
from typing import Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.engine import Tensor
from hypercoil_examples.atlas.ellgat import ELLGAT, ELLGATBlock
from hypercoil_examples.atlas.icosphere import (
    connectivity_matrix, icosphere
)


#TODO: Implement a general version of the ELLGAT-UNet below
# class ELLGATUNet(eqx.Module):
#     blocks: List[ELLGATBlock]
#     nlin: callable = jax.nn.leaky_relu

#     def __init__(
#         self,
#         subdivisions: Tuple[int, ...],
#         channels: Tuple[int, ...],
#         attn_heads: Union[int, Tuple[int, ...]],
#         nlin: callable = jax.nn.leaky_relu,
#         *,
#         key: 'jax.random.PRNGKey',
#     ):
#         n_blocks = len(subdivisions) - 1
#         assert n_blocks == len(channels) - 1
#         if isinstance(attn_heads, int):
#             attn_heads = (attn_heads,) * n_blocks
#         assert n_blocks == len(attn_heads)

#         blocks = []
#         # contractive path
#         for i, heads in enumerate(attn_heads):
#             key_b = jax.random.fold_in(key, i)
#             blocks.append(
#                 ELLGATBlock(
#                     query_features=channels[i],
#                     key_features=channels[i],
#                     out_features=channels[i + 1],
#                     attn_heads=heads,
#                     nlin=nlin,
#                     key=key_b,
#                 )
#             )
#             if subdivisions[i + 1] != subdivisions[i]:
#                 # Bipartite ELLGAT
#                 key_b = jax.random.split(key_b)[0]
#                 blocks.append(
#                     ELLGAT(
#                         query_features=channels[i + 1],
#                         key_features=channels[i + 1],
#                         out_features=channels[i + 1],
#                         attn_heads=heads,
#                         nlin=nlin,
#                         key=key_b,
#                     )
#                 )

#         # expansive path
#         for i, heads in enumerate(attn_heads[::-1]):
#             key_b = jax.random.fold_in(key, n_blocks + i)
#             blocks.append(
#                 ELLGATBlock(
#                     query_features=channels[-i - 1] + channels[-i - 2],
#                     key_features=channels[-i - 1] + channels[-i - 2],
#                     out_features=channels[-i - 2],
#                     attn_heads=heads,
#                     nlin=nlin,
#                     key=key_b,
#                 )
#             )
#             if subdivisions[-i - 2] != subdivisions[-i - 1]:
#                 # Bipartite ELLGAT
#                 key_b = jax.random.split(key_b)[0]
#                 blocks.append(
#                     ELLGAT(
#                         query_features=channels[-i - 2],
#                         key_features=channels[-i - 2],
#                         out_features=channels[-i - 2],
#                         attn_heads=heads,
#                         nlin=nlin,
#                         key=key_b,
#                     )
#                 )
#         self.blocks = blocks

#     def __call__(
#         self,
#         adj: Tensor,
#         x: Tensor,
#         *,
#         key: 'jax.random.PRNGKey',
#     ) -> Tensor:
#         pass


class IcoELLGATUNet(eqx.Module):
    contractive: Tuple[ELLGATBlock]
    expansive: Tuple[ELLGATBlock]
    resample: Mapping[Tuple[int, int], ELLGAT]
    ingress: Mapping[Tuple[int, int], ELLGAT]
    readout: ELLGAT
    icospheres: Tuple[Tensor, ...]
    bipartite: Mapping[Tuple[int, int], Tensor]
    argadj: Mapping[Tuple[int, int], Tensor]
    nlin: callable = jax.nn.leaky_relu

    def __init__(
        self,
        base_coor: Tensor,
        base_mask: Tensor,
        base_adj: Tensor,
        in_dim: Tuple[int, int, int],
        hidden_dim: Tuple[int, int, int],
        resolution: Tuple[int, int, int],
        attn_heads: Tuple[int, int, int],
        readout_dim: int,
        hidden_readout_dim: int,
        ingress_level: Tuple[Optional[int], Optional[int], Optional[int]],
        nlin: callable = jax.nn.leaky_relu,
        max_connections_default: int = 16,
        max_connections_explicit: Optional[
            Mapping[Tuple[int, int], int]
        ] = None,
        *,
        key: 'jax.random.PRNGKey',
    ):
        base_coor = jnp.asarray(base_coor)
        base_mask = jnp.asarray(base_mask)
        num_levels = len(resolution)
        assert num_levels == len(attn_heads)
        assert num_levels == len(in_dim)
        assert num_levels == len(hidden_dim)
        assert num_levels == len(ingress_level)

        icospheres = ((base_coor, None),) + tuple(
            icosphere(r) for r in resolution[1:]
        )
        vertices, icospheres = zip(*icospheres)
        icospheres = (base_adj,) + tuple(
            connectivity_matrix(vertices[i + 1], ico)
            for i, ico in enumerate(icospheres[1:])
        )
        max_connections = defaultdict(lambda: max_connections_default)
        if max_connections_explicit is not None:
            max_connections.update(max_connections_explicit)
        closest = {
            (i, i + 1): (ico_in @ ico_out.T)
            for i, (ico_in, ico_out) in enumerate(
                zip(vertices[:-1], vertices[1:])
            )
        }
        # masks = {0: base_mask}
        for i, (ingress, ico) in enumerate(zip(ingress_level, vertices)):
            if ingress and closest.get((0, i), None) is None:
                closest[(0, i)] = (base_coor @ ico.T)
            # if i > 0:
            #     masks[i] = base_mask[closest[(0, i)].argmax(0)]

        # Mask the closest points and coordinate matrices
        # Note: we don't end up doing this here because it leads to an (even
        # more) uneven number of neighbours for each vertex, which results in
        # a ragged tensor and more unnecessary computation under the ELL
        # format.
        # vertices = tuple(ico[masks[i]] for i, ico in enumerate(vertices))
        # for (i, j), e in closest.items():
        #     closest[(i, j)] = e[masks[i]][:, masks[j]]

        bipartite = {
            (i, j): jax.vmap(
                lambda x: jnp.where(
                    x == e.argmax(1),
                    size=max_connections[(i, j)],
                    fill_value=-1,
                )[0],
                in_axes=(-1,)
            )(
                jnp.arange(vertices[j].shape[0])[None, ...]
            )
            for (i, j), e in closest.items()
        }
        bipartite = {
            (i, j): e[..., ~jnp.all(e == -1, axis=0)]
            for (i, j), e in bipartite.items()
        }
        # Mask the closest points and coordinate matrices
        masks = {0: base_mask}
        for (i, j), e in bipartite.items():
            mask = masks.get((i, j), None)
            if mask is None:
                mask = ~(
                    (e.at[~masks[i][e]].set(-1) == -1).sum(1) == e.shape[-1]
                )
            masks[(i, j)] = mask
            # prioritise masks derived directly from the base mask
            if i == 0:
                masks[j] = mask
        # Now make the masks consistent
        for (i, j), e in bipartite.items():
            ref_mask = masks[j]
            mask = masks[(i, j)]
            # missing = jnp.where(~mask & ref_mask)[0]
            # Shove extra vertices into the closest available downsampled
            # vertex
            extra = jnp.where(mask & ~ref_mask)[0]
            if extra.size > 0:
                rows = bipartite[(i, j)][extra]
                indices = rows[masks[i][rows[extra]].at[rows == -1].set(0)]
                allowed = ref_mask * ((bipartite[(i, j)] == -1).sum(-1) > 0)
                consistent_rows = (
                    closest[(i, j)][indices] + jnp.where(allowed, 0, -jnp.inf)
                ).argmax(-1)
                for row, e in zip(consistent_rows, indices):
                    col = (
                        ~(bipartite[(i, j)][row] == -1)
                    ).argmin() # first -1: an available slot
                    bipartite[(i, j)] = (
                        bipartite[(i, j)].at[row, col].set(indices[0])
                    )
        vertices = tuple(ico[masks[i]] for i, ico in enumerate(vertices))
        cumul = {
            i: (j.cumsum() - 1).at[~masks[i]].set(-1)
            for i, j in masks.items()
        }
        icospheres = tuple(
            cumul[i][e].at[e == -1].set(-1)[masks[i]]
            for i, e in enumerate(icospheres)
        )
        bipartite = {
            (i, j): cumul[i][e].at[e == -1].set(-1)[masks[j]]
            for (i, j), e in bipartite.items()
        }
        argadj = {
            (i, j): jnp.unique(e, return_index=True) + (e.shape[-1],)
            for (i, j), e in bipartite.items()
        }
        argadj = {
            (i, j): e[u != -1] // s
            for (i, j), (u, e, s) in argadj.items()
        }

        contractive = []
        expansive = []
        resample = {}
        ingress = {}
        for i in range(num_levels):
            key = jax.random.fold_in(key, i)
            key_c, key_e = jax.random.split(key)
            contractive_in_dim = in_dim[i]
            expansive_in_extra = (
                0 if (i == num_levels - 1)
                else (hidden_dim[i + 1] * attn_heads[i + 1])
            )
            expansive_out_dim = (
                hidden_dim[i] if i != 0 else hidden_readout_dim
            )
            if ingress_level[i]:
                key_i, key_c = jax.random.split(key_c)
                ingress[i] = ELLGAT(
                    query_features=ingress_level[i],
                    out_features=in_dim[i],
                    attn_heads=attn_heads[i],
                    nlin=nlin,
                    key=key_i,
                )
                #TODO: This should be more flexible
                contractive_in_dim += in_dim[i] * attn_heads[i]
            if (
                (i + 1) < len(resolution) and
                resolution[i] != resolution[i + 1]
            ):
                key_r, key_c = jax.random.split(key_c)
                resample[(i, i + 1)] = ELLGAT(
                    query_features=hidden_dim[i] * attn_heads[i],
                    out_features=in_dim[i + 1] // attn_heads[i + 1],
                    attn_heads=attn_heads[i + 1],
                    nlin=nlin,
                    key=key_r,
                )
            contractive.append(
                ELLGATBlock(
                    query_features=contractive_in_dim,
                    out_features=hidden_dim[i],
                    attn_heads=attn_heads[i],
                    nlin=nlin,
                    key=key_c,
                )
            )
            expansive.append(
                ELLGATBlock(
                    query_features=(
                        hidden_dim[i] * attn_heads[i] + expansive_in_extra
                    ),
                    out_features=expansive_out_dim,
                    attn_heads=attn_heads[i],
                    nlin=nlin,
                    key=key_e,
                )
            )

        key_r = jax.random.fold_in(key, num_levels)
        readout = ELLGAT(
            query_features=hidden_readout_dim * attn_heads[0],
            out_features=readout_dim,
            attn_heads=1,
            nlin=nlin,
            key=key_r,
        )

        self.contractive = tuple(contractive)
        self.expansive = tuple(expansive)
        self.resample = resample
        self.ingress = ingress
        self.readout = readout
        self.icospheres = icospheres
        self.bipartite = bipartite
        self.argadj = argadj
        self.nlin = nlin

    def __call__(
        self,
        X: Tuple[Tensor, ...],
        *,
        key: 'jax.random.PRNGKey',
    ) -> Tensor:
        key_c, key_e = jax.random.split(key)
        Q, X = X[0], X[1:]
        Z = []
        for i, module in enumerate(self.contractive):
            key_i = jax.random.fold_in(key_c, i)
            Qi = None
            if self.ingress.get(i, None) is not None:
                key_i, key_r = jax.random.split(key_i)
                # Prepare scatter-mean as query, and the original Qi as key
                # on a bipartite graph
                Qi, X = X[0], X[1:]
                Qir = jnp.zeros(
                    (Qi.shape[0], self.bipartite[(0, i)].shape[0])
                ).at[..., self.argadj[(0, i)]].add(Qi)
                count = jnp.zeros(
                    (self.bipartite[(0, i)].shape[0],)
                ).at[self.argadj[(0, i)]].add(jnp.ones(Qi.shape[1]))
                Qi = self.ingress[i](
                    adj=self.bipartite[(0, i)],
                    Q=Qir / count,
                    K=Qi,
                    key=key_r,
                )
                Qi = self.nlin(
                    Qi.reshape(
                        *Qi.shape[:-3],
                        Qi.shape[-3] * Qi.shape[-2],
                        Qi.shape[-1],
                    )
                )
            if self.resample.get((i - 1, i), None) is not None:
                key_i, key_r = jax.random.split(key_i)
                # Prepare scatter-mean as query, and the original Q as key
                # on a bipartite graph
                Qr = jnp.zeros(
                    (Q.shape[0], self.bipartite[(i - 1, i)].shape[0])
                ).at[..., self.argadj[(i - 1, i)]].add(Q)
                count = jnp.zeros(
                    (self.bipartite[(i - 1, i)].shape[0],)
                ).at[self.argadj[(i - 1, i)]].add(jnp.ones(Q.shape[1]))
                Q = self.resample[(i - 1, i)](
                    adj=self.bipartite[(i - 1, i)],
                    Q=Qr / count,
                    K=Q,
                    key=key_r,
                )
                Q = self.nlin(
                    Q.reshape(
                        *Q.shape[:-3],
                        Q.shape[-3] * Q.shape[-2],
                        Q.shape[-1],
                    )
                )
            if Qi is not None:
                Q = jnp.concatenate((Q, Qi), axis=0)
            Q = module(
                adj=self.icospheres[i],
                Q=Q,
                key=key_i,
            )
            Q = self.nlin(Q)
            Z.append(Q)

        Q = None
        for i, module in enumerate(self.expansive[::-1]):
            key_i = jax.random.fold_in(key_e, i)
            idx = len(self.expansive) - i - 1
            Z, Qn = Z[:-1], Z[-1]
            if Q is not None:
                Q = jnp.concatenate((Q, Qn), axis=0)
            else:
                Q = Qn
            Q = module(
                adj=self.icospheres[idx],
                Q=Q,
                key=key_i,
            )
            Q = self.nlin(Q)
            adjarg = self.argadj.get((idx - 1, idx), None)
            if adjarg is not None:
                # Gather the Q to the original size
                Q = Q[..., adjarg]

        key_r = jax.random.fold_in(key, len(self.contractive))
        Q = self.readout(
            adj=self.icospheres[0],
            Q=Q,
            key=key_r,
        )
        Q = jax.nn.softmax(
            Q.reshape(*Q.shape[:-3], *Q.shape[-2:]),
            axis=-2,
        )
        return Q


def main():
    import templateflow.api as tflow
    import nibabel as nb
    base_coor_L_path = tflow.get(
        'fsLR', density='32k', hemi='L', space=None, suffix='sphere'
    )
    base_mask_L_path = tflow.get(
        'fsLR', density='32k', hemi='L', desc='nomedialwall'
    )
    base_coor = nb.load(base_coor_L_path).darrays[0].data / 100
    base_mask = nb.load(base_mask_L_path).darrays[0].data.astype(bool)
    base_adj = connectivity_matrix(
        base_coor,
        nb.load(base_coor_L_path).darrays[1].data,
    )

    model = IcoELLGATUNet(
        base_coor=base_coor,
        base_mask=base_mask,
        base_adj=base_adj,
        in_dim=(3, 16, 64),
        hidden_dim=(16, 64, 128),
        hidden_readout_dim=64,
        resolution=(None, 25, 9),
        attn_heads=(4, 4, 4),
        readout_dim=200,
        ingress_level=(None, 16, 64),
        max_connections_explicit={
            (0, 1): 16,
            (0, 2): 64,
            (1, 2): 16,
        },
        key=jax.random.PRNGKey(0),
    )
    Q = (
        jax.random.normal(
            jax.random.PRNGKey(17),
            (
                model.contractive[0].layers[0].query_features,
                model.icospheres[0].shape[0],
            ),
        ),
        jax.random.normal(
            jax.random.PRNGKey(18),
            (
                model.ingress[1].query_features,
                model.icospheres[0].shape[0],
                #model.icospheres[1].shape[0],
            ),
        ),
        jax.random.normal(
            jax.random.PRNGKey(19),
            (
                model.ingress[2].query_features,
                model.icospheres[0].shape[0],
                #model.icospheres[2].shape[0],
            ),
        ),
    )
    result = model(
        Q,
        key=jax.random.PRNGKey(0),
    )
    assert 0


if __name__ == "__main__":
    main()
