#!/usr/bin/env python3
# scripts/generate_mix_40_40_20.py
# 从现有 ALPINE 严格图生成 40/40/20 (t=20) 的 train_20.txt 与 test.txt
# 同时复制 composition_graph.graphml 与 stage_info.pkl 到输出目录

import argparse
import os
import random
import pickle
import shutil
from typing import List, Tuple, Dict, Set

import networkx as nx
import numpy as np


def load_graph_and_stages(graph_dir: str):
    gml = os.path.join(graph_dir, "composition_graph.graphml")
    stg = os.path.join(graph_dir, "stage_info.pkl")
    if not os.path.exists(gml):
        raise FileNotFoundError(f"Missing {gml}")
    if not os.path.exists(stg):
        raise FileNotFoundError(f"Missing {stg}")

    G = nx.read_graphml(gml)
    with open(stg, "rb") as f:
        info = pickle.load(f)
    if "stages" not in info:
        raise ValueError("stage_info.pkl 缺少 'stages' 键")

    S1, S2, S3 = info["stages"]

    # 统一转为字符串集合
    def to_str_set(x):
        return set(str(i) for i in x)

    S1 = to_str_set(S1)
    S2 = to_str_set(S2)
    S3 = to_str_set(S3)

    # 一致性检查
    nodes = set(G.nodes())
    missing = (S1 | S2 | S3) - nodes
    if missing:
        print(f"[WARN] stages 中 {len(missing)} 个节点不在图里，示例: {list(sorted(missing))[:5]}")

    return G, (S1, S2, S3)


class AlpinePathGenerator:
    """
    ALPINE 风格的随机路径生成：
    - 预计算 ancestors(t)
    - 每步仅从仍可达目标的后继中随机选择
    - 多次尝试避免死路
    - 默认不允许重复节点
    """
    def __init__(self, G: nx.DiGraph, allow_revisit: bool = False, max_factor: int = 2):
        self.G = G
        self.allow_revisit = allow_revisit
        self.max_len = max_factor * max(2, G.number_of_nodes())
        self.anc: Dict[str, Set[str]] = {t: nx.ancestors(G, t) for t in G.nodes()}
        self.succ = {u: list(G.successors(u)) for u in G.nodes()}

    def _one_try(self, src: str, tgt: str):
        if src == tgt:
            return [src]
        if src not in self.anc.get(tgt, set()):
            return None

        path = [src]
        cur = src
        visited = set([src]) if not self.allow_revisit else set()

        for _ in range(self.max_len):
            if cur == tgt:
                return path
            candidates = []
            for n in self.succ.get(cur, []):
                if n == tgt or (n in self.anc.get(tgt, set())):
                    if self.allow_revisit or (n not in visited):
                        candidates.append(n)
            if not candidates:
                return None
            nxt = random.choice(candidates)
            path.append(nxt)
            cur = nxt
            if not self.allow_revisit:
                visited.add(nxt)
        return None

    def random_path(self, src: str, tgt: str, must_pass_S2: bool = False, S2: Set[str] = None):
        for _ in range(100):
            p = self._one_try(src, tgt)
            if p is None:
                continue
            if must_pass_S2 and S2 is not None:
                inner = p[1:-1]
                if not any(x in S2 for x in inner):
                    continue
            return p
        return None


def reachable_pairs(G: nx.DiGraph, sources: Set[str], targets: Set[str]) -> List[Tuple[str, str]]:
    pairs = []
    anc = {t: nx.ancestors(G, t) for t in targets}
    for s in sources:
        for t in targets:
            if s != t and (s in anc[t]):
                pairs.append((s, t))
    return pairs


def make_path_pool(
    G: nx.DiGraph,
    pairs: List[Tuple[str, str]],
    gen: AlpinePathGenerator,
    per_pair: int,
    must_pass_S2: bool = False,
    S2: Set[str] = None,
) -> List[List[str]]:
    pool: List[List[str]] = []
    for (u, v) in pairs:
        got = 0
        trials = 0
        need = per_pair
        while got < need and trials < per_pair * 100:
            trials += 1
            p = gen.random_path(u, v, must_pass_S2=must_pass_S2, S2=S2)
            if p is None:
                continue
            rec = [u, v] + p  # 一行：[src tgt path...]；path 从 src 到 tgt
            pool.append(rec)
            got += 1
    return pool


def build_mix(P12, P23, P13, n: int, t_percent: int, seed: int):
    """
    D_t(n): 采样 (1-t%)*n/2 from P12, (1-t%)*n/2 from P23, t%*n from P13
    若路径池不足，使用有放回采样
    """
    rng = random.Random(seed)
    t = t_percent / 100.0
    n12 = int((1 - t) * n / 2)
    n23 = int((1 - t) * n / 2)
    n13 = n - n12 - n23

    def pick(pool, k):
        if k <= 0:
            return []
        if not pool:
            return []
        if len(pool) >= k:
            return rng.sample(pool, k)
        return [pool[rng.randrange(len(pool))] for _ in range(k)]

    D = []
    D.extend(pick(P12, n12))
    D.extend(pick(P23, n23))
    D.extend(pick(P13, n13))
    rng.shuffle(D)
    return D


def write_lines(path: str, seqs: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in seqs:
            f.write(" ".join(str(x) for x in s) + "\n")


def main():
    ap = argparse.ArgumentParser(description="从现有 ALPINE 严格图生成 40/40/20 (t=20) 的训练/验证集")
    ap.add_argument("--graph_dir", required=True, help="包含 composition_graph.graphml 与 stage_info.pkl 的目录")
    ap.add_argument("--out_dir", required=True, help="输出目录（建议 data/simple_graph/mix_alpine_90）")
    ap.add_argument("--dataset_size", type=int, default=10000, help="训练集样本数 n")
    ap.add_argument("--val_size", type=int, default=1000, help="验证集样本数（test.txt）")
    ap.add_argument("--paths_per_pair", type=int, default=20, help="构建路径池时每个可达 pair 生成的路径数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_revisit", action="store_true", help="路径可否重复节点（默认不允许）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("Generate 40/40/20 (t=20) dataset from existing ALPINE graph")
    print("=" * 80)
    print(f"Graph dir      : {args.graph_dir}")
    print(f"Out dir        : {args.out_dir}")
    print(f"Train size (n) : {args.dataset_size}")
    print(f"Val size       : {args.val_size}")
    print(f"paths_per_pair : {args.paths_per_pair}")
    print(f"seed           : {args.seed}")
    print(f"allow_revisit  : {args.allow_revisit}")

    # 1) 读取图与分层
    G, (S1, S2, S3) = load_graph_and_stages(args.graph_dir)
    print(f"Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    print(f"|S1|={len(S1)}, |S2|={len(S2)}, |S3|={len(S3)}")

    # 2) 初始化路径生成器
    gen = AlpinePathGenerator(G, allow_revisit=args.allow_revisit, max_factor=2)

    # 3) 可达对
    pairs_12 = reachable_pairs(G, S1, S2)
    pairs_23 = reachable_pairs(G, S2, S3)
    pairs_13 = reachable_pairs(G, S1, S3)
    print(f"Reachable pairs: S1→S2={len(pairs_12)}, S2→S3={len(pairs_23)}, S1→S3={len(pairs_13)}")

    # 4) 路径池（S1→S3 强制经过 S2）
    print("\nBuilding path pools...")
    P12 = make_path_pool(G, pairs_12, gen, args.paths_per_pair, must_pass_S2=False, S2=None)
    P23 = make_path_pool(G, pairs_23, gen, args.paths_per_pair, must_pass_S2=False, S2=None)
    P13 = make_path_pool(G, pairs_13, gen, args.paths_per_pair, must_pass_S2=True, S2=S2)
    print(f"Path pool sizes: P12={len(P12)}, P23={len(P23)}, P13={len(P13)}")

    # 5) 构造 40/40/20 训练/验证集
    os.makedirs(args.out_dir, exist_ok=True)
    D20 = build_mix(P12, P23, P13, n=args.dataset_size, t_percent=20, seed=args.seed + 20)
    V20 = build_mix(P12, P23, P13, n=args.val_size, t_percent=20, seed=args.seed + 1020)

    write_lines(os.path.join(args.out_dir, "train_20.txt"), D20)
    write_lines(os.path.join(args.out_dir, "test.txt"), V20)
    print(f"\nWritten: {os.path.join(args.out_dir, 'train_20.txt')} ({len(D20)})")
    print(f"Written: {os.path.join(args.out_dir, 'test.txt')} ({len(V20)})")

    # 6) 复制图与阶段信息到输出目录（训练脚本会用到）
    for fname in ["composition_graph.graphml", "stage_info.pkl"]:
        src = os.path.join(args.graph_dir, fname)
        dst = os.path.join(args.out_dir, fname)
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    # 7) 写一份简单的元信息
    meta = {
        "graph_dir": args.graph_dir,
        "paths_per_pair_for_pool": args.paths_per_pair,
        "train_size": args.dataset_size,
        "val_size": args.val_size,
        "t_percent": 20,
        "seed": args.seed,
        "allow_revisit": args.allow_revisit,
        "pool_sizes": {"P12": len(P12), "P23": len(P23), "P13": len(P13)},
    }
    with open(os.path.join(args.out_dir, "mix_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()