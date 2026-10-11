# -*- coding: utf-8 -*-
""" Build a CA-SUM-format h5 file (+ splits json) from a random subsample of the MoSu dataset (TripleSumm).

By default the files are read remotely from the Hugging Face hub with HTTP range requests, so only the sampled videos
are transferred (the ~40GB feature files are not downloaded). Use --local_dir if you already have the files on disk.

Example:
    python tools/make_mosu_subset.py --n_videos 200 --modalities visual audio text
"""
import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np

REPO = "datasets/hminjeong/TripleSumm-MoSu"
FEATURE_FILES = {"visual": "mosu_feat_visual_clip.h5",
                 "audio": "mosu_feat_audio_ast.h5",
                 "text": "mosu_feat_text_roberta.h5"}
GT_FILE = "mosu_gt.h5"
SPLIT_FILE = "mosu_split.json"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CASUM_DATA_DIR", ROOT / "data"))


def parse_args():
    parser = argparse.ArgumentParser("Create a CA-SUM compatible subsample of MoSu")
    parser.add_argument("--n_videos", type=int, default=200, help="Number of videos in the subsample")
    parser.add_argument("--pool", type=str, default="all", choices=["train", "val", "test", "all"],
                        help="Official MoSu split(s) to sample the videos from")
    parser.add_argument("--modalities", type=str, nargs="+", default=["visual"], choices=list(FEATURE_FILES),
                        help="Modalities to copy. 'visual' is stored as `features` (what CA-SUM reads), the rest as "
                             "`features_audio` / `features_text`")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of random train/test splits to create")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--local_dir", type=str, default=None, help="Read MoSu files from this directory instead of HF")
    parser.add_argument("--out_h5", type=str, default=str(DATA_DIR / "MoSu/mosu_subset.h5"))
    parser.add_argument("--out_splits", type=str, default=str(DATA_DIR / "splits/mosu_splits.json"))
    return parser.parse_args()


class Source:
    """ Opens MoSu files either from a local directory or remotely from the Hugging Face hub. """

    def __init__(self, local_dir):
        self.local_dir = Path(local_dir) if local_dir else None
        self.fs, self.paths = None, {}
        if self.local_dir is None:
            from huggingface_hub import HfFileSystem
            self.fs = HfFileSystem()
            # Locate files anywhere inside the repo (they may sit in a sub-folder)
            for path in self.fs.find(REPO):
                self.paths.setdefault(path.rsplit("/", 1)[-1], path)

    def open(self, name, mode="rb"):
        if self.local_dir is not None:
            return open(self.local_dir / name, mode)
        if name not in self.paths:
            raise FileNotFoundError(f"{name} not found in {REPO}. Available: {sorted(self.paths)}")
        return self.fs.open(self.paths[name], mode, block_size=1 << 20)

    def h5(self, name):
        return h5py.File(self.open(name), "r")


def to_shots(change_points, n):
    """ Return shots as [start, end] (inclusive) rows clipped to [0, n-1], as in the eccv16 TVSum/SumMe files. """
    cps = np.asarray(change_points).astype(np.int64)
    if cps.ndim == 1:                                   # boundary list -> [start, end] rows
        bounds = np.unique(np.concatenate([[0], cps, [n]]))
        bounds = bounds[(bounds >= 0) & (bounds <= n)]
        cps = np.stack([bounds[:-1], bounds[1:] - 1], axis=1)
    elif len(cps) > 1 and np.all(cps[1:, 0] == cps[:-1, 1]):
        cps[:, 1] -= 1                                  # exclusive ends -> inclusive ends
    cps[:, 1] = np.minimum(cps[:, 1], n - 1)
    return cps[cps[:, 1] >= cps[:, 0]]


def main():
    args = parse_args()
    src = Source(args.local_dir)

    with src.open(SPLIT_FILE, "r") as handle:
        official = json.load(handle)
    pools = ["train", "val", "test"] if args.pool == "all" else [args.pool]
    candidates = sorted({key for p in pools for key in official[f"{p}_keys"]})

    rng = random.Random(args.seed)
    sampled = rng.sample(candidates, min(args.n_videos, len(candidates)))
    print(f"Sampled {len(sampled)} of {len(candidates)} videos from the '{args.pool}' pool")

    feat_h5 = {m: src.h5(FEATURE_FILES[m]) for m in args.modalities}
    gt_h5 = src.h5(GT_FILE)
    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    try:
        with h5py.File(out_path, "w") as out:
            for i, vid in enumerate(sampled, start=1):
                if any(vid not in f for f in feat_h5.values()) or vid not in gt_h5:
                    print(f"  [skip] {vid}: missing in a feature or gt file")
                    continue
                feats = {m: np.asarray(f[vid], dtype=np.float32) for m, f in feat_h5.items()}
                gt_score = np.asarray(gt_h5[vid]["gt_score"], dtype=np.float32).reshape(-1)
                gt_summary = np.asarray(gt_h5[vid]["gt_summary"]).reshape(-1).astype(np.int8)

                lengths = [len(x) for x in feats.values()] + [len(gt_score), len(gt_summary)]
                n = min(lengths)
                if max(lengths) - n > 2:
                    print(f"  [warn] {vid}: length mismatch {lengths}, truncating to {n}")
                if n < 2:
                    print(f"  [skip] {vid}: too short ({n} steps)")
                    continue
                shots = to_shots(gt_h5[vid]["change_points"], n)

                grp = out.create_group(vid)
                main_mod = "visual" if "visual" in feats else args.modalities[0]
                grp["features"] = feats[main_mod][:n]
                for m, x in feats.items():
                    if m != main_mod:
                        grp[f"features_{m}"] = x[:n]
                grp["gtscore"] = gt_score[:n]
                grp["user_summary"] = gt_summary[:n][None, :]   # single "annotator"
                grp["gtsummary"] = gt_summary[:n]
                grp["change_points"] = shots
                grp["n_frame_per_seg"] = shots[:, 1] - shots[:, 0] + 1
                grp["n_frames"] = n                              # MoSu is 1 step per second: no sub-sampling
                grp["n_steps"] = n
                grp["picks"] = np.arange(n, dtype=np.int64)
                grp["video_name"] = vid
                kept.append(vid)
                print(f"  [{i}/{len(sampled)}] {vid}: {n} steps, {len(shots)} shots")
    finally:
        for f in feat_h5.values():
            f.close()
        gt_h5.close()

    splits = []
    for s in range(args.n_splits):
        keys = kept[:]
        random.Random(args.seed + s).shuffle(keys)
        n_train = int(round(len(keys) * args.train_ratio))
        splits.append({"train_keys": keys[:n_train], "test_keys": keys[n_train:]})
    Path(args.out_splits).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_splits, "w") as handle:
        json.dump(splits, handle)

    print(f"Wrote {len(kept)} videos to {out_path} and {args.n_splits} splits to {args.out_splits}")


if __name__ == "__main__":
    main()
