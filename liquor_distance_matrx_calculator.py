#!/usr/bin/env python3
import pandas as pd
import numpy as np

# ─────────── PARAMETERS ───────────
# (none needed for pure distance matrix)

# ─────────── 1) LOAD YOUR TAXONOMY ───────────
# CSV: two columns, no header: child,parent
#also do for Fruithut_taxonomy_data.csv
tax = pd.read_csv(
    'liquor_taxonomy.csv',
    header=None,
    names=['child','parent'],
    dtype=int
)

# Build child→parent map
parent_map = {row.child: row.parent for row in tax.itertuples()}

# Any parent that never appears as a child is a root → map to None
all_parents  = set(parent_map.values())
all_children = set(parent_map.keys())
for root in all_parents - all_children:
    parent_map[root] = None

# ─────────── 2) IDENTIFY LEAF NODES ───────────
# All 4‑digit codes are leaves
leaf_nodes = sorted(n for n in parent_map if len(str(n)) == 5)

# ─────────── 3) BUILD ANCESTOR PATHS (INCLUDING None) ───────────
ancestor_paths = {}
ancestor_index = {}

for leaf in leaf_nodes:
    path = []
    cur = leaf
    while True:
        path.append(cur)
        par = parent_map[cur]
        if par is None:
            path.append(None)
            break
        cur = par

    ancestor_paths[leaf] = path
    ancestor_index[leaf] = {anc: dist for dist, anc in enumerate(path)}

# ─────────── 4) ALLOCATE & FILL DISTANCE MATRIX ───────────
n = len(leaf_nodes)
D = np.zeros((n, n), dtype=int)

for i_idx, i in enumerate(leaf_nodes):
    path_i = ancestor_paths[i]
    for j_idx in range(i_idx + 1, n):
        j = leaf_nodes[j_idx]
        idx_map_j = ancestor_index[j]

        # tree‐distance = min(dist(i,anc) + dist(j,anc)) over common ancestors
        d_ij = min(
            i_dist + idx_map_j[anc]
            for i_dist, anc in enumerate(path_i)
            if anc in idx_map_j
        )

        D[i_idx, j_idx] = D[j_idx, i_idx] = d_ij

# diagonal is zero by definition
np.fill_diagonal(D, 0)

# ─────────── 5) SAVE AS CSV ───────────
D_df = pd.DataFrame(D, index=leaf_nodes, columns=leaf_nodes)
D_df.to_csv('distance_matrix_liquor.csv')

print(f"✅ Done! Saved distance_matrix.csv ({n}×{n})")
