"""検出器の比較（#15）：track.py の tracks.csv を、MOT の正解枠（選手・GK 22人）と画像上で照合する。

使い方:
    python eval_detect.py --mot data/mot/118577.txt --tracks outputs/cmp_rfdetr/tracks.csv outputs/cmp_yolox/tracks.csv ...

- 照合はフレームごとに IoU のハンガリアン法で1対1にし、IoU 0.5 以上を一致とみなす。
- 正解は選手・GK だけ（審判などは含まない）ため、適合率は低めに出る。比較は同じ条件の相対値として見る。
- tracks.csv は ID が確定した枠だけなので、検出器そのものではなく「検出＋追跡」の出力を測っている。
- --pitch-tracks（pitch.py の出力。--tracks と同じ順）を渡すと、ピッチ座標でのフレーム間の揺れも測る。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MATCH = 0.5
LONG_TRACK_FRAMES = 300  # 15秒のうち12秒以上続いた ID を「ほぼ全区間」とみなす（RESULT.md の集計と同じ）


def load_mot(path: Path) -> dict[int, list]:
    """MOT 形式（frame,id,x,y,w,h,...。frame は0始まり）を、フレームごとの (id, xyxy) にする。"""
    by_frame: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        c = line.split(",")
        x, y, w, h = map(float, c[2:6])
        by_frame[int(c[0])].append((int(c[1]), (x, y, x + w, y + h)))
    return by_frame


def load_tracks(path: Path) -> dict[int, list]:
    by_frame: dict[int, list] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_frame[int(r["frame_index"])].append(
                (int(r["track_id"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    return by_frame


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def evaluate(gt: dict[int, list], pred: dict[int, list]) -> dict:
    frames = range(min(pred), max(pred) + 1)
    n_gt = n_pred = n_hit = 0
    ious = []
    ids_per_gt: dict[int, set] = defaultdict(set)  # 正解1人に対応した予測 ID の集合（ID の付け直しの目安）
    length: dict[int, int] = defaultdict(int)
    for k in frames:
        g, p = gt.get(k, []), pred.get(k, [])
        n_gt += len(g)
        n_pred += len(p)
        for tid, _ in p:
            length[tid] += 1
        if not g or not p:
            continue
        m = iou(np.array([b for _, b in p]), np.array([b for _, b in g]))
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= IOU_MATCH
        n_hit += int(ok.sum())
        ious.extend(m[r, c][ok].tolist())
        for pi, gi in zip(r[ok], c[ok]):
            ids_per_gt[g[gi][0]].add(p[pi][0])
    per_gt = [len(v) for v in ids_per_gt.values()]
    return {
        "frames": len(frames),
        "recall": round(n_hit / n_gt, 3),  # 延べ（フレーム×正解の選手）のうち IoU 0.5 以上で対応した割合
        "precision_vs_players": round(n_hit / n_pred, 3) if n_pred else None,  # 審判・観客なども分母に入る
        "matched_iou_median": round(float(np.median(ious)), 3) if ious else None,
        "pred_boxes_per_frame": round(n_pred / len(frames), 1),
        "unique_ids": len(length),
        "ids_over_300_frames": sum(v >= LONG_TRACK_FRAMES for v in length.values()),
        "ids_per_gt_player_mean": round(float(np.mean(per_gt)), 2) if per_gt else None,
        "ids_per_gt_player_max": max(per_gt) if per_gt else None,
    }


def pitch_jitter(path: Path) -> dict:
    """同じ ID の連続フレーム間の移動量（m）。ピッチ内の行だけを使う（REPORT.md §5.3 の集計と同じ）。"""
    by_id: dict[int, list] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["in_pitch"] == "1":
                by_id[int(r["track_id"])].append((int(r["frame_index"]), float(r["x_m"]), float(r["y_m"])))
    steps = []
    for rows in by_id.values():
        a = np.array(sorted(rows))
        consecutive = np.diff(a[:, 0]) == 1
        steps.append(np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1)[consecutive])
    st = np.concatenate(steps)
    return {"step_m_median": round(float(np.median(st)), 3), "step_m_p90": round(float(np.percentile(st, 90)), 3),
            "step_over_0_5m_ratio": round(float((st > 0.5).mean()), 3)}  # 0.5m/フレーム = 12.5m/s


def main() -> None:
    p = argparse.ArgumentParser(description="tracks.csv を MOT の正解枠と画像上で照合する")
    p.add_argument("--mot", required=True, help="mot/<試合>.txt")
    p.add_argument("--tracks", required=True, nargs="+", help="track.py の tracks.csv（複数可）")
    p.add_argument("--pitch-tracks", nargs="+", help="pitch.py の pitch_tracks.csv（--tracks と同じ順）")
    p.add_argument("--output", help="結果の JSON を書く先（省略時は表示だけ）")
    args = p.parse_args()
    if args.pitch_tracks and len(args.pitch_tracks) != len(args.tracks):
        p.error("--pitch-tracks は --tracks と同じ数だけ渡す")

    gt = load_mot(Path(args.mot))
    result = {str(Path(t).parent.name): evaluate(gt, load_tracks(Path(t))) for t in args.tracks}
    for t, pt in zip(args.tracks, args.pitch_tracks or []):
        result[str(Path(t).parent.name)]["pitch_jitter"] = pitch_jitter(Path(pt))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
