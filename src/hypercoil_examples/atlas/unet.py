# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ELLGAT u-net
~~~~~~~~~~~~
A u-net model composed of ELLGAT blocks. We draw considerable inspiration from
SUGAR, a GAT-based u-net model for brain surface registration. The code for
this model is available at
https://github.com/IndiLab/SUGAR/blob/main/models/gatunet_model.py
"""
from collections import defaultdict
from typing import Literal, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.engine import Tensor
from hypercoil_examples.atlas.ellgat import ELLGAT, ELLGATBlock
from hypercoil_examples.atlas.icosphere import (
    connectivity_matrix, icosphere
)


def scatter_mean_bipartite(
    module: ELLGAT,
    Q: Tensor,
    argadj: Tensor,
    bipartite: Tensor,
    nlin: callable,
    *,
    inference: Optional[bool] = None,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    # Prepare scatter-mean as query, and the original Q as key
    # on a bipartite graph
    Qr = jnp.zeros(
        (Q.shape[0], bipartite.shape[0])
    ).at[..., argadj].add(Q)
    count = jnp.zeros(
        (bipartite.shape[0],)
    ).at[argadj].add(jnp.ones(Q.shape[1]))
    Qr = jnp.where(count == 0, 0, Qr)
    count = jnp.where(count == 0, 1, count)
    Q = module(
        adj=bipartite,
        Q=Qr / count,
        K=Q,
        inference=inference,
        key=key,
    )
    Q = nlin(
        Q.reshape(
            *Q.shape[:-3],
            Q.shape[-3] * Q.shape[-2],
            Q.shape[-1],
        )
    )
    return Q


class ELLMesh(eqx.Module):
    resolution: Tuple[int, ...]
    icospheres: Tuple[Tensor, ...]
    bipartite: Mapping[Tuple[int, int], Tensor]
    argadj: Mapping[Tuple[int, int], Tensor]
    ingress_level: Tuple[Optional[int], ...]

    def __init__(
        self,
        base_coor: Tensor,
        base_adj: Tensor,
        resolution: Tuple[int, ...],
        ingress_level: Tuple[Optional[int], ...],
        base_mask: Optional[Tensor] = None,
        max_connections_default: int = 16,
        max_connections_explicit: Optional[
            Mapping[Tuple[int, int], int]
        ] = None,
        add_self_connections: bool = True,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        base_coor = jnp.asarray(base_coor)
        base_adj = jnp.asarray(base_adj)
        if base_mask is None:
            base_mask = jnp.ones(base_coor.shape[0], dtype=bool)
        else:
            base_mask = jnp.asarray(base_mask)

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
            missing = jnp.where(~mask & ref_mask)[0]
            # Shove extra vertices into the closest available downsampled
            # vertex
            extra = jnp.where(mask & ~ref_mask)[0]
            if extra.size > 0:
                rows = bipartite[(i, j)][extra]
                indices = rows[masks[i][rows].at[rows == -1].set(0)]
                allowed = ref_mask * ((bipartite[(i, j)] == -1).sum(-1) > 0)
                consistent_rows = (
                    closest[(i, j)][indices] + jnp.where(allowed, 0, -jnp.inf)
                ).argmax(-1)
                #if missing: assert 0
                for row, e in zip(consistent_rows, indices):
                    col = (
                        ~(bipartite[(i, j)][row] == -1)
                    ).argmin() # first -1: an available slot
                    bipartite[(i, j)] = (
                        bipartite[(i, j)].at[row, col].set(e)
                    )
        vertices = tuple(ico[masks[i]] for i, ico in enumerate(vertices))
        cumul = {
            i: (j.cumsum() - 1).at[~j].set(-1)
            for i, j in masks.items()
        }
        #if missing: assert 0
        icospheres = tuple(
            cumul[i][e].at[e == -1].set(-1)[masks[i]]
            for i, e in enumerate(icospheres)
        )
        if add_self_connections:
            icospheres = tuple(
                jnp.concatenate(
                    (jnp.arange(ico.shape[0])[..., None], ico),
                    axis=-1,
                )
                for ico in icospheres
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
        # if len(argadj[(1, 2)]) != len(bipartite[(0, 1)]):
        #     assert 0

        self.resolution = resolution
        self.icospheres = icospheres
        self.bipartite = bipartite
        self.argadj = argadj
        self.ingress_level = ingress_level


class IcoELLGATUNet(eqx.Module):
    contractive: Tuple[ELLGATBlock]
    expansive: Tuple[ELLGATBlock]
    resample: Mapping[Tuple[int, int], ELLGAT]
    ingress: Mapping[Tuple[int, int], ELLGAT]
    readout: ELLGAT
    meshes: Mapping[str, ELLMesh]
    nlin: callable = jax.nn.leaky_relu
    norm: Optional[eqx.Module] = None

    def __init__(
        self,
        meshes: Mapping[str, ELLMesh],
        in_dim: Tuple[int, int, int],
        hidden_dim: Tuple[int, int, int],
        attn_heads: Tuple[int, int, int],
        readout_dim: int,
        hidden_readout_dim: int,
        nlin: callable = jax.nn.leaky_relu,
        norm: Optional[eqx.Module] = None,
        dropout: Optional[float] = None,
        dropout_inference: bool = False,
        readout_skip_dim: int = 0,
        *,
        key: 'jax.random.PRNGKey',
    ):
        mesh_names = tuple(meshes.keys())
        resolution = meshes[mesh_names[0]].resolution
        ingress_level = meshes[mesh_names[0]].ingress_level
        num_levels = len(resolution)
        assert num_levels == len(attn_heads)
        assert num_levels == len(in_dim)
        assert num_levels == len(hidden_dim)
        for mesh in meshes.values():
            assert num_levels == len(mesh.resolution)
            assert num_levels == len(mesh.ingress_level)

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
                    dropout=dropout,
                    dropout_inference=dropout_inference,
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
                    dropout=dropout,
                    dropout_inference=dropout_inference,
                    key=key_r,
                )
            contractive.append(
                ELLGATBlock(
                    query_features=contractive_in_dim,
                    out_features=hidden_dim[i],
                    attn_heads=attn_heads[i],
                    nlin=nlin,
                    norm=norm,
                    dropout=dropout,
                    dropout_inference=dropout_inference,
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
                    norm=norm,
                    dropout=dropout,
                    dropout_inference=dropout_inference,
                    key=key_e,
                )
            )

        key_r = jax.random.fold_in(key, num_levels)
        readout = ELLGAT(
            query_features=hidden_readout_dim * attn_heads[0] + readout_skip_dim,
            out_features=readout_dim,
            attn_heads=1,
            nlin=nlin,
            dropout=dropout,
            dropout_inference=dropout_inference,
            key=key_r,
        )

        self.meshes = meshes
        self.contractive = tuple(contractive)
        self.expansive = tuple(expansive)
        self.resample = resample
        self.ingress = ingress
        self.readout = readout
        self.nlin = nlin
        if norm is None:
            norm = eqx.nn.Identity()
        self.norm = norm

    def __call__(
        self,
        X: Tuple[Tensor, ...],
        mesh: Optional[str] = None,
        *,
        inference: Optional[bool] = None,
        key: 'jax.random.PRNGKey',
    ) -> Tensor:
        if mesh is None:
            assert len(self.meshes) == 1
            mesh = next(iter(self.meshes))
        else:
            mesh = self.meshes[mesh]

        key_c, key_e = jax.random.split(key)
        Q, X = X[0], X[1:]
        Z = []
        for i, module in enumerate(self.contractive):
            key_i = jax.random.fold_in(key_c, i)
            Qi = None
            if self.ingress.get(i, None) is not None:
                key_i, key_r = jax.random.split(key_i)
                Qi, X = X[0], X[1:]
                Qi = scatter_mean_bipartite(
                    module=self.ingress[i],
                    Q=Qi,
                    argadj=mesh.argadj[(0, i)],
                    bipartite=mesh.bipartite[(0, i)],
                    nlin=self.nlin,
                    inference=inference,
                    key=key_r,
                )
                Qi = self.norm(Qi)
            if self.resample.get((i - 1, i), None) is not None:
                key_i, key_r = jax.random.split(key_i)
                Q = scatter_mean_bipartite(
                    module=self.resample[(i - 1, i)],
                    Q=Q,
                    argadj=mesh.argadj[(i - 1, i)],
                    bipartite=mesh.bipartite[(i - 1, i)],
                    nlin=self.nlin,
                    inference=inference,
                    key=key_r,
                )
                Q = self.norm(Q)
            if Qi is not None:
                Q = jnp.concatenate((Q, Qi), axis=-2)
            Q = module(
                adj=mesh.icospheres[i],
                Q=Q,
                inference=inference,
                key=key_i,
            )
            Q = self.nlin(Q)
            Q = self.norm(Q)
            Z.append(Q)

        Q = None
        for i, module in enumerate(self.expansive[::-1]):
            key_i = jax.random.fold_in(key_e, i)
            idx = len(self.expansive) - i - 1
            Z, Qn = Z[:-1], Z[-1]
            if Q is not None:
                Q = jnp.concatenate((Q, Qn), axis=-2)
            else:
                Q = Qn
            Q = module(
                adj=mesh.icospheres[idx],
                Q=Q,
                inference=inference,
                key=key_i,
            )
            Q = self.nlin(Q)
            Q = self.norm(Q)
            adjarg = mesh.argadj.get((idx - 1, idx), None)
            if adjarg is not None:
                # Gather the Q to the original size
                Q = Q[..., adjarg]
            # if (
            #     (Q is not None and jnp.any(jnp.isnan(Q))) or
            #     (Qi is not None and jnp.any(jnp.isnan(Qi)))
            # ):
            #     assert 0

        key_r = jax.random.fold_in(key, len(self.contractive))
        if len(X) > 0 and X[0] is not None:
            Qi, X = X[0], X[1:]
            Q = jnp.concatenate((Q, Qi), axis=-2)
        Q = self.readout(
            adj=mesh.icospheres[0],
            Q=Q,
            inference=inference,
            key=key_r,
        )
        if len(X) > 0 and X[0] is not None:
            # Residual learning
            Qi, X = X[0], X[1:]
            Q = Q + Qi
        Q = jax.nn.softmax(
            Q.reshape(*Q.shape[:-3], *Q.shape[-2:]),
            axis=-2,
        )
        return Q

def get_base_coor_mask_adj(hemi: str) -> Tuple[Tensor, Tensor, Tensor]:
    import templateflow.api as tflow
    import nibabel as nb
    base_coor_path = tflow.get(
        'fsLR', density='32k', hemi=hemi, space=None, suffix='sphere'
    )
    base_mask_path = tflow.get(
        'fsLR', density='32k', hemi=hemi, desc='nomedialwall'
    )
    base_coor = nb.load(base_coor_path).darrays[0].data / 100
    base_mask = nb.load(base_mask_path).darrays[0].data.astype(bool)
    base_adj = connectivity_matrix(
        base_coor,
        nb.load(base_coor_path).darrays[1].data,
    )
    return base_coor, base_mask, base_adj


def get_meshes(
    model: Literal['test', 'full'] = 'test',
    positional_dim: Optional[int] = None,
) -> Tuple[ELLMesh, ELLMesh]:

    def get_mesh(hemi: str) -> ELLMesh:
        base_coor, base_mask, base_adj = get_base_coor_mask_adj(hemi)
        if model == 'test':
            ingress_level = (None, 16, 64)
        elif model == 'full':
            ingress_level = (None, 64, 658)
        if positional_dim is not None:
            ingress_level = tuple(
                i + positional_dim if i is not None else i
                for i in ingress_level
            )
        return ELLMesh(
            base_coor=base_coor,
            base_adj=base_adj,
            resolution=(None, 25, 9),
            ingress_level=ingress_level,
            base_mask=base_mask,
            max_connections_explicit={
                (0, 1): 16,
                (0, 2): 64,
                (1, 2): 16,
            },
            key=jax.random.PRNGKey(0),
        )

    mesh_L = get_mesh('L')
    mesh_R = get_mesh('R')
    return mesh_L, mesh_R


def main(visualise: bool = False):
    mesh_L, mesh_R = get_meshes()

    if visualise:
        from hyve import (
            plotdef,
            surf_from_archive,
            surf_scalars_from_array,
            plot_to_image,
            save_grid,
        )
        plot_f = plotdef(
            surf_from_archive(),
            surf_scalars_from_array('pools', is_masked=True),
            plot_to_image(),
            save_grid(
                n_cols=8,
                n_rows=1,
                padding=10,
                canvas_size=(3200, 300),
                canvas_color=(0, 0, 0),
                fname_spec='ellgatunet_{hemi}_{surfscalars}.png',
                sort_by=['surfscalars'],
                scalar_bar_action='collect',
            ),
        )
        for k in mesh_L.argadj:
            array_L = mesh_L.argadj[k]
            array_R = mesh_R.argadj[k]
            k_orig = k
            while k[0] != 0:
                array_L = array_L[mesh_L.argadj[k[0] - 1, k[0]]]
                array_R = array_R[mesh_R.argadj[k[0] - 1, k[0]]]
                k = (k[0] - 1, k[1])
            plot_f(
                template='fsLR',
                load_mask=True,
                pools_array_left=array_L,
                pools_array_right=array_R,
                surf_scalars_cmap='prism',
                surf_projection=('veryinflated',),
                hemisphere=['left', 'right', None],
                views={
                    'left': ('medial', 'lateral'),
                    'right': ('medial', 'lateral'),
                    'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
                },
                output_dir='/tmp',
                fname_spec=f'scalars-{k_orig[0]}x{k_orig[1]}',
                window_size=(800, 600),
            )

    model = IcoELLGATUNet(
        meshes={
            'cortex_L': mesh_L,
            'cortex_R': mesh_R,
        },
        in_dim=(3, 16, 64),
        hidden_dim=(16, 64, 128),
        hidden_readout_dim=64,
        attn_heads=(4, 4, 4),
        readout_dim=200,
        dropout=0.6,
        key=jax.random.PRNGKey(0),
    )
    selected_mesh = ('cortex_R', mesh_R)
    Q = (
        jax.random.normal(
            jax.random.PRNGKey(17),
            (
                model.contractive[0].layers[0].query_features,
                selected_mesh[1].icospheres[0].shape[0],
            ),
        ),
        jax.random.normal(
            jax.random.PRNGKey(18),
            (
                model.ingress[1].query_features,
                selected_mesh[1].icospheres[0].shape[0],
                #selected_mesh[1].icospheres[1].shape[0],
            ),
        ),
        jax.random.normal(
            jax.random.PRNGKey(19),
            (
                model.ingress[2].query_features,
                selected_mesh[1].icospheres[0].shape[0],
                #selected_mesh[1].icospheres[2].shape[0],
            ),
        ),
    )
    result = model(
        Q,
        mesh=selected_mesh[0],
        key=jax.random.PRNGKey(0),
    )
    assert 0


if __name__ == "__main__":
    main(visualise=True)
