#!/usr/bin/env python3
# scripts/prepare_mix_alpine_bin.py
# 将 data/simple_graph/mix_alpine_90 下的 train_20.txt 与 test.txt 编码为 .bin
# 生成：train_20.bin、val.bin、meta.pkl（与训练脚本严格兼容）

import os
import pickle
import numpy as np
import argparse


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(total_nodes: int):
    # 与你现有 prepare 脚本一致：0=[PAD], 1='\n', 数字节点从 2 开始
    stoi = {}
    itos = {}

    for i in range(total_nodes):
        stoi[str(i)] = i + 2
        itos[i + 2] = str(i)

    stoi['[PAD]'] = 0
    itos[0] = '[PAD]'
    stoi['\n'] = 1
    itos[1] = '\n'

    vocab_size = total_nodes + 2
    return stoi, itos, vocab_size


def encode_line_to_ids(line: str, stoi: dict):
    # 行内用空格分隔的 token（数字），末尾追加换行符 1
    tokens = line.strip().split(" ")
    ids = []
    for tok in tokens:
        if tok in stoi:
            ids.append(stoi[tok])
    ids.append(1)  # newline
    return ids


def get_block_size(text: str, stoi: dict):
    max_len = 0
    for line in text.splitlines():
        if line.strip() == "":
            continue
        ids = encode_line_to_ids(line, stoi)
        if len(ids) > max_len:
            max_len = len(ids)
    # 向上取整到 32 的倍数，兼容训练脚本
    block_size = ((max_len // 32) + 1) * 32 if max_len % 32 != 0 else max_len
    return block_size


def process_text_to_tokens(text: str, stoi: dict, block_size: int):
    tokens = []
    for line in text.splitlines():
        if line.strip() == "":
            continue
        ids = encode_line_to_ids(line, stoi)
        # 补齐到 block_size
        if len(ids) < block_size:
            ids = ids + [0] * (block_size - len(ids))
        tokens.extend(ids)
    return np.array(tokens, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser(description="Encode mix_alpine 40/40/20 数据为 .bin")
    ap.add_argument("--base_dir", required=True, help="例如 data/simple_graph/mix_alpine_90")
    ap.add_argument("--total_nodes", type=int, default=90, help="节点总数（默认 90）")
    args = ap.parse_args()

    base_dir = args.base_dir
    train_txt = os.path.join(base_dir, "train_20.txt")
    test_txt = os.path.join(base_dir, "test.txt")

    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"Missing {train_txt}")
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"Missing {test_txt}")

    # 读取数据
    train_data = read_text(train_txt)
    val_data = read_text(test_txt)
    print(f"Train dataset length (chars): {len(train_data):,}")
    print(f"Val dataset length   (chars): {len(val_data):,}")

    # 构建词表
    stoi, itos, vocab_size = build_vocab(args.total_nodes)
    print(f"Vocab size: {vocab_size} (Token-level)")

    # 计算 block_size
    block_size = max(get_block_size(train_data, stoi), get_block_size(val_data, stoi))
    # 与原脚本一致做 32 对齐
    if block_size % 32 != 0:
        block_size = ((block_size // 32) + 1) * 32
    print(f"Block size: {block_size}")

    # 编码
    train_ids = process_text_to_tokens(train_data, stoi, block_size)
    val_ids = process_text_to_tokens(val_data, stoi, block_size)
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens  : {len(val_ids):,}")

    # 写出 .bin
    train_bin = os.path.join(base_dir, "train_20.bin")
    val_bin = os.path.join(base_dir, "val.bin")
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)
    print(f"Written: {train_bin}")
    print(f"Written: {val_bin}")

    # 保存元信息（训练脚本需要）
    meta = {
        'unreachable': False,
        'simple_format': True,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"Written: {os.path.join(base_dir, 'meta.pkl')}")

    print("\n✅ Encoding complete.")


if __name__ == "__main__":
    main()