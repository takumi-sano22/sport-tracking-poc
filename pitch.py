"""track.py の tracks.csv（画面座標）を、ピッチを上から見た座標（m）へ変換するPoC。

使い方:
    python pitch.py --tracks outputs/demo/tracks.csv --calib data/raw/118577 \
        --video data/mot/clips/118577.mp4 --start 0 --output outputs/demo_pitch

座標系は SoccerTrack v2 の規約に合わせる：105m×68m、原点はセンターサークル、
x はメインカメラから見て右向き、y はカメラ側のタッチライン向きが正。
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
IN_PITCH_MARGIN = 2.0  # ラインから外側この距離までを in_pitch とする（m）
# キーポイント残差の補正の滑らかさ（0で全点を厳密に通る）。キーポイントはライン上にしかなく、
# 小さい値では左奥の大きな残差の補正がピッチ内部へ波及して悪化した。正解（gsr）のクリップ前半2分で選んだ値
TPS_SMOOTHING = 3000.0
SMOOTH_FRAMES = 5  # 距離・速度の計算前にかける移動平均の幅（検出の揺れで距離が膨らむのを抑える）
TRACE_LENGTH = 30  # 俯瞰図に描く軌跡のフレーム数
PX_PER_M = 10  # 俯瞰図の縮尺
MARGIN_M = 5.0  # 俯瞰図でピッチの外側に描く余白（m）
CSV_HEADER = ["frame_index", "timestamp_sec", "track_id", "x_m", "y_m", "in_pitch", "confidence"]


def fail(message: str) -> None:
    print(f"エラー: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="追跡結果をピッチ座標（俯瞰xy, m）へ変換する")
    parser.add_argument("--tracks", required=True, help="track.py が出力した tracks.csv")
    parser.add_argument("--calib", required=True, help="SoccerTrack v2 の raw/<試合ID> ディレクトリ")
    parser.add_argument("--video", help="元動画（指定するとピッチのラインを重ねた確認画像を出す）")
    parser.add_argument("--start", type=float, default=0.0, help="track.py に渡した --start（俯瞰動画の開始をそろえる）")
    parser.add_argument("--output", required=True, help="出力ディレクトリ（空または未作成）")
    return parser.parse_args()


class PitchMapper:
    """画面のピクセル座標 ⇔ ピッチ座標（m、隅原点）の変換。

    配布元のキャリブレーション（魚眼の歪み補正→ホモグラフィ）で一次変換し、
    手作業で付けられたキーポイントとの残差を薄板スプライン（TPS）で補正する。
    """

    def __init__(self, calib_dir: Path):
        match_id = calib_dir.name
        intr = np.load(calib_dir / f"{match_id}_camera_intrinsics.npz")  # allow_pickle は使わない
        # rvecs/tvecs は pickle のオブジェクト配列のため読まない（ダウンロードしたデータの pickle はコード実行の危険がある）
        self.K, self.D, self.Knew = intr["K"], intr["D"], intr["Knew"]
        self.H = np.load(calib_dir / f"{match_id}_homography.npy")  # ピッチ(m, 隅原点) → 歪み補正後の画素
        self.H_inv = np.linalg.inv(self.H)
        keypoints = json.loads((calib_dir / f"{match_id}_keypoints.json").read_text())
        # キーの "(x,y)" はピッチ座標（m、隅原点）、値は元映像の画素
        self.kp_pitch = np.array([[float(v) for v in re.findall(r"[-\d.]+", k)] for k in keypoints])
        self.kp_image = np.array(list(keypoints.values()), dtype=float)
        base = self._base(self.kp_image)
        self.correction = RBFInterpolator(base, self.kp_pitch - base, kernel="thin_plate_spline",
                                          smoothing=TPS_SMOOTHING)

    def _base(self, image_xy: np.ndarray) -> np.ndarray:
        """配布元のキャリブレーションだけで画素→ピッチ（隅原点）へ変換する。"""
        und = cv2.fisheye.undistortPoints(image_xy.reshape(-1, 1, 2).astype(np.float64), self.K, self.D, P=self.Knew)
        return cv2.perspectiveTransform(und, self.H_inv).reshape(-1, 2)

    def to_pitch(self, image_xy: np.ndarray) -> np.ndarray:
        """画素 → ピッチ座標（m、センター原点）。"""
        if len(image_xy) == 0:
            return np.empty((0, 2))
        base = self._base(image_xy)
        corner = base + self.correction(base)
        return corner - np.array([PITCH_LENGTH / 2, PITCH_WIDTH / 2])

    def to_image(self, pitch_corner_xy: np.ndarray) -> np.ndarray:
        """ピッチ（隅原点）→ 画素。確認用のライン描画に使う（TPS補正は含まない配布元の変換）。"""
        und = cv2.perspectiveTransform(pitch_corner_xy.reshape(-1, 1, 2).astype(np.float64), self.H).reshape(-1, 2)
        norm = np.ascontiguousarray((np.c_[und, np.ones(len(und))] @ np.linalg.inv(self.Knew).T)[:, :2])
        return cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def leave_one_out(self) -> dict:
        """キーポイントを1点ずつ抜いて当てはめ直し、抜いた点の誤差（m）を返す。"""
        base_all = self._base(self.kp_image)
        raw = np.linalg.norm(base_all - self.kp_pitch, axis=1)
        loo = []
        for i in range(len(self.kp_pitch)):
            m = np.arange(len(self.kp_pitch)) != i
            f = RBFInterpolator(base_all[m], self.kp_pitch[m] - base_all[m], kernel="thin_plate_spline",
                                smoothing=TPS_SMOOTHING)
            loo.append(np.linalg.norm(base_all[i] + f(base_all[i:i + 1])[0] - self.kp_pitch[i]))
        loo = np.array(loo)

        def stats(e):
            return {"median": round(float(np.median(e)), 2), "p90": round(float(np.percentile(e, 90)), 2),
                    "max": round(float(e.max()), 2)}
        return {"keypoints": len(loo), "official_only_m": stats(raw), "official_plus_tps_loo_m": stats(loo)}


def pitch_lines() -> list[np.ndarray]:
    """ピッチの主なライン（m、隅原点）を折れ線の点列で返す。"""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    segs = [[(0, 0), (L, 0)], [(L, 0), (L, W)], [(L, W), (0, W)], [(0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
    for x0, x1 in ((0, 16.5), (L, L - 16.5)):  # ペナルティエリア
        segs += [[(x0, 13.84), (x1, 13.84)], [(x1, 13.84), (x1, 54.16)], [(x1, 54.16), (x0, 54.16)]]
    for x0, x1 in ((0, 5.5), (L, L - 5.5)):  # ゴールエリア
        segs += [[(x0, 24.84), (x1, 24.84)], [(x1, 24.84), (x1, 43.16)], [(x1, 43.16), (x0, 43.16)]]
    t = np.linspace(0, 1, 50)[:, None]
    lines = [np.array(a) * (1 - t) + np.array(b) * t for a, b in segs]
    th = np.linspace(0, 2 * np.pi, 100)
    lines.append(np.c_[L / 2 + 9.15 * np.cos(th), W / 2 + 9.15 * np.sin(th)])
    return lines


def write_overlay(mapper: PitchMapper, video: Path, path: Path) -> None:
    """元映像の1フレーム目にピッチのラインとキーポイントを重ねる（キャリブレーションの目視確認用）。"""
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        fail(f"確認画像用に動画を読めません: {video}")
    for line in pitch_lines():
        cv2.polylines(frame, [mapper.to_image(line).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
    for x, y in mapper.kp_image:
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)
    cv2.imwrite(str(path), frame)


def to_canvas(xy_center: np.ndarray) -> np.ndarray:
    """ピッチ座標（センター原点）→ 俯瞰図の画素。"""
    x = (xy_center[..., 0] + PITCH_LENGTH / 2 + MARGIN_M) * PX_PER_M
    y = (xy_center[..., 1] + PITCH_WIDTH / 2 + MARGIN_M) * PX_PER_M
    return np.stack([x, y], axis=-1)


def pitch_background() -> np.ndarray:
    w = int((PITCH_LENGTH + 2 * MARGIN_M) * PX_PER_M)
    h = int((PITCH_WIDTH + 2 * MARGIN_M) * PX_PER_M)
    canvas = np.full((h + h % 2, w + w % 2, 3), (60, 120, 40), np.uint8)  # H.264 のため偶数サイズにする
    for line in pitch_lines():
        pts = to_canvas(line - np.array([PITCH_LENGTH / 2, PITCH_WIDTH / 2]))
        cv2.polylines(canvas, [pts.astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def id_color(track_id: int) -> tuple[int, int, int]:
    """IDごとに見分けやすい色（HSVの色相を黄金角で回す）。"""
    hue = int((track_id * 137.508) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def write_topview(rows_by_frame: dict, frames: range, fps: float, path: Path) -> None:
    """俯瞰図の動画を書く。まず mp4v で書き、ブラウザ用に H.264 へ変換する。"""
    background = pitch_background()
    h, w = background.shape[:2]
    raw_path = path.with_name("topview_mp4v.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        fail(f"俯瞰動画を作れません: {raw_path}")
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRACE_LENGTH))
    for f in frames:
        canvas = background.copy()
        for tid, xy, inside in rows_by_frame.get(f, []):
            history[tid].append((f, tuple(to_canvas(np.array(xy)).astype(int))))
        # 軌跡：フレームが連続する区間だけ結ぶ
        for tid, pts in history.items():
            recent = [(fi, p) for fi, p in pts if f - fi < TRACE_LENGTH]
            for (f0, p0), (f1, p1) in zip(recent, recent[1:]):
                if f1 - f0 == 1:
                    cv2.line(canvas, p0, p1, id_color(tid), 2, cv2.LINE_AA)
        for tid, xy, inside in rows_by_frame.get(f, []):
            p = tuple(to_canvas(np.array(xy)).astype(int))
            color = id_color(tid) if inside else (150, 150, 150)  # ピッチ外は灰色で区別
            cv2.circle(canvas, p, 7, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 7, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{tid}", (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {f}  {f / fps:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()

    # ブラウザ再生用に .venv の imageio-ffmpeg で H.264 へ変換する（sudo・システムFFmpeg不要）
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(raw_path), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", "-an", str(path)]
    if subprocess.run(cmd).returncode != 0:
        fail("俯瞰動画の H.264 変換に失敗しました")
    raw_path.unlink()


def summarize(rows: list[dict], fps: float) -> list[dict]:
    """ID別の走行距離・速度（参考値）。IDが付け直されるため、1人分の値ではない。"""
    by_id: dict[int, list] = defaultdict(list)
    for r in rows:
        by_id[r["track_id"]].append(r)
    out = []
    one_sec = int(round(fps))
    for tid, rs in sorted(by_id.items()):
        rs.sort(key=lambda r: r["frame_index"])
        frames = np.array([r["frame_index"] for r in rs])
        xy = np.array([[r["x_m"], r["y_m"]] for r in rs])
        # フレームが連続する区間ごとに移動平均をかけ、区間内の移動距離だけを足す
        distance, max_speed = 0.0, 0.0
        breaks = np.where(np.diff(frames) != 1)[0] + 1
        for seg_f, seg_xy in zip(np.split(frames, breaks), np.split(xy, breaks)):
            k = min(SMOOTH_FRAMES, len(seg_xy))
            kernel = np.ones(k) / k
            sm = np.c_[np.convolve(seg_xy[:, 0], kernel, "valid"), np.convolve(seg_xy[:, 1], kernel, "valid")]
            distance += float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())
            if len(sm) > one_sec:  # 1秒間の変位から最高速度を出す（フレーム間の揺れの影響を抑える）
                max_speed = max(max_speed, float(np.linalg.norm(sm[one_sec:] - sm[:-one_sec], axis=1).max()))
        duration = len(rs) / fps
        out.append({
            "track_id": tid, "frames": len(rs), "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
            "in_pitch_ratio": round(float(np.mean([r["in_pitch"] for r in rs])), 2),
            "distance_m": round(distance, 1),
            "mean_speed_mps": round(distance / duration, 2) if duration > 0 else 0.0,
            "max_speed_1s_mps": round(max_speed, 2) if max_speed else "",
        })
    return out


def main() -> None:
    args = parse_args()
    tracks_path, calib_dir, output_dir = Path(args.tracks), Path(args.calib), Path(args.output)
    if not tracks_path.is_file():
        fail(f"tracks.csv がありません: {tracks_path}")
    if not calib_dir.is_dir():
        fail(f"キャリブレーションのディレクトリがありません: {calib_dir}（README の取得手順を参照）")
    if args.video and not Path(args.video).is_file():
        fail(f"動画がありません: {args.video}")
    if output_dir.exists() and any(output_dir.iterdir()):
        fail(f"出力先が空ではありません: {output_dir}（別の名前を指定してください）")
    try:
        mapper = PitchMapper(calib_dir)
    except (OSError, KeyError, ValueError) as e:
        fail(f"キャリブレーションを読めません: {e}")

    with open(tracks_path, newline="") as f:
        src = list(csv.DictReader(f))
    if not src:
        fail("tracks.csv にID付きの行がありません")
    # fps は CSV の frame_index / timestamp_sec から求める（track.py は CFR 前提で時刻を書いている）
    sample = next(r for r in src if float(r["timestamp_sec"]) > 0)
    fps = round(int(sample["frame_index"]) / float(sample["timestamp_sec"]), 3)
    # 俯瞰動画は track.py の出力と同じ開始フレームから書く（出力先を作る前に検証する）
    first = int(round(args.start * fps))
    if first > min(int(r["frame_index"]) for r in src):
        fail("--start が tracks.csv の最初のフレームより後です")

    # 足元の点（枠の下辺の中央）をまとめてピッチ座標へ変換する
    feet = np.array([[(float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"])] for r in src])
    xy = mapper.to_pitch(feet)
    half = np.array([PITCH_LENGTH / 2 + IN_PITCH_MARGIN, PITCH_WIDTH / 2 + IN_PITCH_MARGIN])
    inside = np.all(np.abs(xy) <= half, axis=1)

    rows = [{"frame_index": int(r["frame_index"]), "timestamp_sec": r["timestamp_sec"], "track_id": int(r["track_id"]),
             "x_m": float(p[0]), "y_m": float(p[1]), "in_pitch": int(ins), "confidence": r["confidence"]}
            for r, p, ins in zip(src, xy, inside)]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pitch_tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([r["frame_index"], r["timestamp_sec"], r["track_id"], f"{r['x_m']:.2f}", f"{r['y_m']:.2f}",
                        r["in_pitch"], r["confidence"]])

    summary = summarize(rows, fps)
    with open(output_dir / "pitch_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    check = mapper.leave_one_out()
    (output_dir / "calibration_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2))
    if args.video:
        write_overlay(mapper, Path(args.video), output_dir / "calib_overlay.png")

    # 俯瞰動画は開始フレームから CSV の最後のフレームまでを書く
    by_frame: dict[int, list] = defaultdict(list)
    for r in rows:
        by_frame[r["frame_index"]].append((r["track_id"], (r["x_m"], r["y_m"]), r["in_pitch"]))
    last = max(by_frame)
    write_topview(by_frame, range(first, last + 1), fps, output_dir / "topview.mp4")

    n_in = int(inside.sum())
    print(f"fps: {fps} / 行: {len(rows)}（in_pitch {n_in}, ピッチ外 {len(rows) - n_in}） / ID: {len(summary)}")
    print(f"キーポイント検証（m）: 配布元のみ {check['official_only_m']} / TPS補正の1点抜き {check['official_plus_tps_loo_m']}")
    print(f"俯瞰動画: フレーム {first}〜{last}（{last - first + 1}フレーム）")
    print(f"出力: {output_dir}/pitch_tracks.csv, pitch_summary.csv, calibration_check.json, topview.mp4"
          + (", calib_overlay.png" if args.video else ""))


if __name__ == "__main__":
    main()
