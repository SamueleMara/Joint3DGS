from __future__ import annotations

from dataclasses import dataclass

from articulation.data.dataclasses import MatchBatch


@dataclass(frozen=True)
class ObservationNode:
    t: int
    v: int
    k: int


class DisjointSet:
    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_node_index(keypoint_counts: dict[tuple[int, int], int]) -> tuple[list[ObservationNode], dict[ObservationNode, int]]:
    nodes: list[ObservationNode] = []
    index: dict[ObservationNode, int] = {}

    for (t, v), n in sorted(keypoint_counts.items()):
        for k in range(int(n)):
            node = ObservationNode(t=int(t), v=int(v), k=int(k))
            index[node] = len(nodes)
            nodes.append(node)
    return nodes, index


def match_to_edges(
    match: MatchBatch,
    frame_a: tuple[int, int],
    frame_b: tuple[int, int],
) -> list[tuple[ObservationNode, ObservationNode, float]]:
    ta, va = frame_a
    tb, vb = frame_b

    if match.idx_a.shape[0] != match.idx_b.shape[0] or match.idx_a.shape[0] != match.confidence.shape[0]:
        raise ValueError("match idx/confidence size mismatch")

    edges: list[tuple[ObservationNode, ObservationNode, float]] = []
    for i in range(match.idx_a.shape[0]):
        a = ObservationNode(t=int(ta), v=int(va), k=int(match.idx_a[i].item()))
        b = ObservationNode(t=int(tb), v=int(vb), k=int(match.idx_b[i].item()))
        c = float(match.confidence[i].item())
        edges.append((a, b, c))
    return edges


def connected_components(
    nodes: list[ObservationNode],
    index: dict[ObservationNode, int],
    edges: list[tuple[ObservationNode, ObservationNode, float]],
) -> list[list[ObservationNode]]:
    dsu = DisjointSet(len(nodes))
    for a, b, _ in edges:
        ia = index.get(a)
        ib = index.get(b)
        if ia is None or ib is None:
            continue
        dsu.union(ia, ib)

    comp_map: dict[int, list[ObservationNode]] = {}
    for i, node in enumerate(nodes):
        r = dsu.find(i)
        comp_map.setdefault(r, []).append(node)

    return list(comp_map.values())
