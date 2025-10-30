# routing/simplify.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple

def simplify_path_gc(
    path_idx: List[int],
    nodes: List[Tuple[float,float]],
    visible_fn,
    COLLISION_PREP_M,
    land_tree,
    max_passes: int = 8
) -> List[Tuple[float,float]]:
    """
    迭代「可視直連」簡化：
      - 只要 i 與 j 可視，中間點就捨去
      - 從最遠 j 往回試，能直連就跳
    回傳：簡化後座標序列（lon,lat）
    """
    if not path_idx or len(path_idx) < 2:
        return [nodes[i] for i in path_idx]

    pts = [nodes[i] for i in path_idx]
    for _ in range(max_passes):
        changed = False
        new_pts = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            jumped = False
            for j in range(len(pts) - 1, i + 1, -1):
                if visible_fn(pts[i], pts[j], COLLISION_PREP_M, land_tree):
                    if j > i + 1:
                        new_pts.append(pts[j])
                        i = j
                        changed = True
                        jumped = True
                        break
            if not jumped:
                new_pts.append(pts[i + 1])
                i += 1
        if len(new_pts) >= 2 and new_pts[-1] == new_pts[-2]:
            new_pts.pop()
        pts = new_pts
        if not changed:
            break
    return pts
