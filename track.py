"""最小サッカー追跡PoC: RF-DETR Small（COCO）で人物検出 → ByteTrack → 枠・ID・軌跡付きMP4とCSV。

使い方:
    python track.py --input data/mot/clips/118577.mp4 --start 0 --duration 3 --output outputs/smoke
"""

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
from trackers import ByteTrackTracker

# PoC用の初期値（精度の根拠はない。IMPLEMENTATION.md §4）
DETECTION_THRESHOLD = 0.10  # これ未満の検出は捨てる。低信頼側もByteTrackの2段目照合へ渡すため低めにする
HIGH_CONF_THRESHOLD = 0.25  # ByteTrackの高信頼／低信頼の境界
TRACK_ACTIVATION_THRESHOLD = 0.25  # 新規IDを作る最低スコア（既定0.7では小さい人物にIDが付きにくい）
LOST_TRACK_BUFFER = 30  # 見失ってもIDを保持する長さ（trackersでは30fps換算。25fpsでも約1秒）
TRACE_LENGTH = 30  # 軌跡として描く直近フレーム数
CSV_HEADER = ["frame_index", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "confidence"]


def fail(message: str) -> None:
    """短いエラーを出して終了する。"""
    print(f"エラー: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR Small + ByteTrack による人物追跡PoC")
    parser.add_argument("--input", required=True, help="入力MP4")
    parser.add_argument("--start", type=float, default=0.0, help="開始秒（元クリップ先頭から）")
    parser.add_argument("--duration", type=float, default=15.0, help="処理する秒数")
    parser.add_argument("--output", required=True, help="出力ディレクトリ（空または未作成）")
    return parser.parse_args()


def person_class_id() -> int:
    """COCO重みの predict が返す class_id は COCO_CLASSES のキー（0始まりの class_names とは別）。"""
    ids = [k for k, v in COCO_CLASSES.items() if v == "person"]
    if len(ids) != 1:
        fail("COCOクラス表から person のIDを特定できません")
    return ids[0]


def draw_traces(frame: np.ndarray, history: dict, frame_index: int) -> None:
    """ID別に直近TRACE_LENGTHフレームの下辺中央を結ぶ。フレームが飛んだ区間は線を引かない。"""
    for track_id, points in history.items():
        recent = [(f, p) for f, p in points if frame_index - f < TRACE_LENGTH]
        for (f0, p0), (f1, p1) in zip(recent, recent[1:]):
            if f1 - f0 == 1:
                color = sv.ColorPalette.DEFAULT.by_idx(track_id).as_bgr()
                cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # 入力・引数・出力先の検証（重いモデル読込より先に行う）
    if not input_path.is_file():
        fail(f"入力ファイルがありません: {input_path}")
    if args.start < 0 or args.duration <= 0:
        fail("--start は0以上、--duration は0より大きい値にしてください")
    if output_dir.exists() and any(output_dir.iterdir()):
        fail(f"出力先が空ではありません: {output_dir}（別の名前を指定してください）")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        fail(f"動画を開けません: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0:
        fail("fps またはフレーム数を取得できません")

    # 秒指定を元クリップのフレーム番号へ変換（CFR前提で frame_index / fps を時刻の基準にする）
    start_frame = int(round(args.start * fps))
    n_frames = max(1, int(round(args.duration * fps)))
    if start_frame >= total_frames:
        fail(f"--start がクリップ長（{total_frames / fps:.2f}秒）を超えています")
    end_frame = min(start_frame + n_frames, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if actual_start != start_frame:
        fail(f"開始フレームへ移動できません（要求 {start_frame} / 実際 {actual_start}）")

    # モデルと追跡器は1回だけ作り、全フレームを同じインスタンスへ順に渡す。
    # 読込失敗で中途半端な出力先を残さないよう、出力先の作成より前に読み込む
    t0 = time.perf_counter()
    model = RFDETRSmall(device="cpu")
    load_sec = time.perf_counter() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "annotated.mp4"
    csv_path = output_dir / "tracks.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        fail(f"出力動画を作れません: {video_path}")

    print(f"入力: {input_path} {width}x{height} {fps:.3f}fps 全{total_frames}フレーム")
    print(f"区間: フレーム {start_frame}〜{end_frame - 1}（{end_frame - start_frame}フレーム）")

    person_id = person_class_id()
    tracker = ByteTrackTracker(
        lost_track_buffer=LOST_TRACK_BUFFER,
        frame_rate=fps,
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        high_conf_det_threshold=HIGH_CONF_THRESHOLD,
    )
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=3, color_lookup=sv.ColorLookup.TRACK)
    pending_annotator = sv.BoxAnnotator(thickness=1, color=sv.Color.from_hex("#9e9e9e"))
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRACE_LENGTH))

    csv_rows = 0
    ids_seen: set[int] = set()
    frames_done = 0
    t1 = time.perf_counter()
    try:
        with open(csv_path, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(CSV_HEADER)
            for frame_index in range(start_frame, end_frame):
                ok, frame = cap.read()
                if not ok:
                    print(f"警告: フレーム {frame_index} で読込が終わりました", file=sys.stderr)
                    break
                timestamp = frame_index / fps

                # OpenCVはBGR、RF-DETRはRGB入力
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(rgb, threshold=DETECTION_THRESHOLD)
                detections = detections[detections.class_id == person_id]
                # 空の検出も渡して追跡器の時間を進める
                tracked = tracker.update(detections)

                has_id = tracked.tracker_id >= 0
                confirmed = tracked[has_id]
                pending = tracked[~has_id]

                for (x1, y1, x2, y2), tid, conf in zip(confirmed.xyxy, confirmed.tracker_id, confirmed.confidence):
                    out.writerow([frame_index, f"{timestamp:.4f}", int(tid),
                                  f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{conf:.4f}"])
                    history[int(tid)].append((frame_index, (int((x1 + x2) / 2), int(y2))))
                    ids_seen.add(int(tid))
                    csv_rows += 1

                # ID未確定は細い灰色枠のみ（ラベルなし）で区別する
                canvas = pending_annotator.annotate(frame.copy(), pending)
                draw_traces(canvas, history, frame_index)
                canvas = box_annotator.annotate(canvas, confirmed)
                canvas = label_annotator.annotate(
                    canvas, confirmed, labels=[f"#{t}" for t in confirmed.tracker_id])
                writer.write(canvas)
                frames_done += 1

                if frames_done % 25 == 0:
                    elapsed = time.perf_counter() - t1
                    print(f"  {frames_done}/{end_frame - start_frame} フレーム（{elapsed:.1f}秒）", flush=True)
    except OSError as e:
        fail(f"書込に失敗しました: {e}")
    finally:
        cap.release()
        writer.release()
    process_sec = time.perf_counter() - t1

    # OpenCVで書けても読めるとは限らないため、開き直してフレーム数を確認する
    check = cv2.VideoCapture(str(video_path))
    reread_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if check.isOpened() else 0
    reread_fps = check.get(cv2.CAP_PROP_FPS) if check.isOpened() else 0.0
    check.release()

    print(f"モデル読込: {load_sec:.1f}秒 / 処理: {process_sec:.1f}秒"
          f"（{process_sec / max(frames_done, 1):.2f}秒/フレーム）")
    print(f"処理フレーム: {frames_done}（要求 {end_frame - start_frame}） / 再読込フレーム: {reread_frames}"
          f" / 再読込fps: {reread_fps:.3f} / CSV行: {csv_rows} / ID数: {len(ids_seen)}")
    print(f"出力: {video_path} , {csv_path}")
    if frames_done == 0:
        fail("フレームを1枚も読み込めませんでした")
    if reread_frames != frames_done:
        fail("出力動画のフレーム数が処理数と一致しません")


if __name__ == "__main__":
    main()
