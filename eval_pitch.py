"""ピッチ座標の変換を、SoccerTrack v2 の正解（gsr のピッチ座標）と比べる評価スクリプト。

使い方:
    python eval_pitch.py --gsr data/gsr/118577/118577_1st.json --mot data/mot/118577.txt \
        --calib data/raw/118577 --pitch-tracks outputs/demo_pitch/pitch_tracks.csv --output outputs/demo_eval

1. gsr（数GBのJSON）から、フレーム・選手・ピッチ座標だけをストリームで抜き出して .npz にキャッシュする。
2. MOT の正解枠（クリップの全フレーム・22人）の足元をピッチ座標へ変換し、gsr と最も一致する
   フレームのずれを探して、クリップが試合のどの区間かを特定する（区間はどこにも記録されていないため）。
3. 誤差を2種類測る。
   - 変換の誤差：正解枠の足元 → 変換 → gsr の座標（検出器の誤りを含まない）
   - 全体の誤差：pitch.py の出力（検出・追跡を含む） → gsr の座標
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from scipy.optimize import linear_sum_assignment

from pitch import PitchMapper, fail

MATCH_RADIUS = 2.0  # 予測と正解をこの距離（m）以内なら同一人物とみなす（検出の再現率・適合率用）
ALIGN_SAMPLE_FRAMES = 12  # 区間特定に使うクリップのフレーム数（全体に等間隔で取る）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ピッチ座標の変換を gsr の正解と比べる")
    p.add_argument("--gsr", required=True, help="gsr/<試合>/<試合>_1st.json などの正解JSON")
    p.add_argument("--mot", required=True, help="mot/<試合>.txt（クリップの正解枠）")
    p.add_argument("--calib", required=True, help="raw/<試合> のキャリブレーション")
    p.add_argument("--pitch-tracks", help="pitch.py の pitch_tracks.csv（全体の誤差を測る場合）")
    p.add_argument("--output", required=True, help="出力ディレクトリ（空または未作成）")
    return p.parse_args()


def load_gsr(path: Path) -> dict:
    """gsr の JSON から (frame, track_id, role, x, y) を抜き出す。2回目以降は .npz キャッシュを使う。"""
    cache = path.with_suffix(".extract.npz")
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    frames, tids, roles, xs, ys = [], [], [], [], []
    with open(path, "rb") as f:
        # 全体を読み込むとメモリが足りないため、annotations を1件ずつ読む
        for a in ijson.items(f, "annotations.item", use_float=True):
            bp = a.get("bbox_pitch") or {}
            x, y = bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
            if x is None or y is None:
                continue
            frames.append(int(str(a["image_id"])[-6:]))  # image_id の末尾6桁が動画のフレーム番号（1始まり）
            tids.append(int(a["track_id"]))
            roles.append(str((a.get("attributes") or {}).get("role", "")))
            xs.append(x)
            ys.append(y)
    data = {"frame": np.array(frames, np.int32), "track_id": np.array(tids, np.int32),
            "role": np.array(roles), "x": np.array(xs, np.float32), "y": np.array(ys, np.float32)}
    np.savez(cache, **data)
    return data


def load_mot(path: Path) -> dict[int, np.ndarray]:
    """MOT 形式（frame,id,x,y,w,h,...）から、フレームごとの足元の画素（枠の下辺の中央）を返す。"""
    by_frame: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        c = line.split(",")
        frame, x, y, w, h = int(c[0]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
        by_frame[frame].append((x + w / 2, y + h))
    return {k: np.array(v) for k, v in by_frame.items()}


def matched_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """ハンガリアン法で1対1に対応づけ、対応した組の距離を返す。"""
    if len(pred) == 0 or len(gt) == 0:
        return np.empty(0)
    cost = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return cost[r, c]


def stats(e: np.ndarray) -> dict:
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "mean": round(float(e.mean()), 2), "median": round(float(np.median(e)), 2),
            "p90": round(float(np.percentile(e, 90)), 2), "max": round(float(e.max()), 2)}


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    for p in (args.gsr, args.mot):
        if not Path(p).is_file():
            fail(f"ファイルがありません: {p}")
    if out.exists() and any(out.iterdir()):
        fail(f"出力先が空ではありません: {out}")

    mapper = PitchMapper(Path(args.calib))
    gsr = load_gsr(Path(args.gsr))
    players = np.isin(gsr["role"], ["player", "goalkeeper"])  # MOT の正解は選手とGKの22人だけ
    gt_by_frame: dict[int, np.ndarray] = {}
    order = np.argsort(gsr["frame"], kind="stable")
    fr, xs, ys, pl = gsr["frame"][order], gsr["x"][order], gsr["y"][order], players[order]
    bounds = np.r_[0, np.where(np.diff(fr) != 0)[0] + 1, len(fr)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        m = pl[s:e]
        gt_by_frame[int(fr[s])] = np.c_[xs[s:e][m], ys[s:e][m]]
    print(f"gsr: {len(gsr['frame'])} 件 / フレーム {min(gt_by_frame)}〜{max(gt_by_frame)}")

    mot = load_mot(Path(args.mot))
    mot_pitch = {k: mapper.to_pitch(v) for k, v in mot.items()}
    clip_frames = sorted(mot_pitch)
    samples = [clip_frames[int(i)] for i in np.linspace(0, len(clip_frames) - 1, ALIGN_SAMPLE_FRAMES)]

    # クリップのフレーム k が gsr のフレーム k+offset に当たるとして、全オフセットで一致度を測る
    gsr_frames = np.array(sorted(gt_by_frame))
    best = []
    for offset in range(int(gsr_frames.min()) - samples[0], int(gsr_frames.max()) - samples[-1] + 1):
        errs = [np.median(matched_distances(mot_pitch[k], gt_by_frame[k + offset]))
                for k in samples if (k + offset) in gt_by_frame]
        if len(errs) == len(samples):
            best.append((float(np.median(errs)), offset))
    best.sort()
    if not best:
        fail("クリップと gsr の区間を照合できませんでした")
    score, offset = best[0]
    runner_up = next((round(s, 2) for s, o in best if abs(o - offset) > 25), None)
    print(f"区間の特定: オフセット {offset}（一致度 中央値 {score:.2f} m、1秒以上離れた次点 {runner_up} m）")

    # 変換の誤差：正解枠の足元を変換した位置と gsr の位置を比べる。
    # TPS の滑らかさはクリップ前半で選んだため、選定に使っていない後半も分けて示す
    half = np.array([52.5, 34.0])
    mid = clip_frames[len(clip_frames) // 2]

    def conv_error(frames, use_tps):
        d = [matched_distances(mot_pitch[k] if use_tps else mapper._base(mot[k]) - half, gt_by_frame[k + offset])
             for k in frames if (k + offset) in gt_by_frame]
        return stats(np.concatenate(d))
    first, second = [k for k in clip_frames if k < mid], [k for k in clip_frames if k >= mid]
    result = {"gsr_file": Path(args.gsr).name, "clip_to_gsr_frame_offset": offset,
              "clip_start_in_half_sec": round((offset - 1) / 25.0, 2),  # クリップのフレーム0 = gsr のフレーム offset（1始まり）
              "align_median_m": round(score, 2),
              "align_runner_up_median_m": runner_up,
              "conversion_error_m": {
                  "official_plus_tps": {"clip_first_half": conv_error(first, True), "clip_second_half": conv_error(second, True)},
                  "official_only": {"clip_first_half": conv_error(first, False), "clip_second_half": conv_error(second, False)}}}

    # 全体の誤差：pitch.py の出力（in_pitch の行）と gsr を比べる
    if args.pitch_tracks:
        pred_by_frame: dict[int, list] = defaultdict(list)
        with open(args.pitch_tracks, newline="") as f:
            for r in csv.DictReader(f):
                if r["in_pitch"] == "1":
                    pred_by_frame[int(r["frame_index"])].append((float(r["x_m"]), float(r["y_m"])))
        dists, n_gt, n_pred, n_hit = [], 0, 0, 0
        for k, pts in pred_by_frame.items():
            gt = gt_by_frame.get(k + offset)
            if gt is None:
                continue
            d = matched_distances(np.array(pts), gt)
            n_gt += len(gt)
            n_pred += len(pts)
            n_hit += int((d <= MATCH_RADIUS).sum())
            dists.append(d[d <= MATCH_RADIUS])
        result["end_to_end"] = {
            "frames": len(pred_by_frame), "match_radius_m": MATCH_RADIUS,
            "recall": round(n_hit / n_gt, 3) if n_gt else None,  # 正解22人のうち2m以内に予測がある割合
            "precision": round(n_hit / n_pred, 3) if n_pred else None,  # 予測（ピッチ内）のうち正解と対応した割合
            "matched_error_m": stats(np.concatenate(dists)) if dists else {"n": 0},
            "note": "予測には審判・誤検出を含むため、適合率は低めに出る。gsrの座標は1.05m×0.68m刻み。"}

    out.mkdir(parents=True, exist_ok=True)
    (out / "pitch_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
