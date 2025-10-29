# routing/simplify.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple
from .visibility import Visibility

def simplify_path_gc(
    path_idx: List[int],
    nodes: List[Tuple[float,float]],
    visibility: Visibility,
    max_passes: int = 8
) -> List[Tuple[float,float]]:
    if not path_idx or len(path_idx) < 2:
        return [nodes[i] for i in path_idx]
    pts = [nodes[i] for i in path_idx]
    passes=0
    while passes < max_passes:
        passes += 1
        changed=False
        new_pts=[pts[0]]
        i=0
        while i < len(pts)-1:
            jumped=False
            for j in range(len(pts)-1, i+1, -1):
                if visibility.is_visible(pts[i], pts[j]):
                    if j > i+1:
                        new_pts.append(pts[j]); i=j; changed=True; jumped=True; break
            if not jumped:
                new_pts.append(pts[i+1]); i+=1
        if len(new_pts) >= 2 and new_pts[-1] == new_pts[-2]:
            new_pts.pop()
        pts=new_pts
        if not changed: break
    return pts
