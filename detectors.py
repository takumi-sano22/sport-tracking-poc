"""比較用の人物検出器（#15）。どれも RGB の1フレームを受け取り、person だけの sv.Detections を返す。

既定は RF-DETR Small（track.py の本来の構成）。それ以外は比較のためだけに使う。
すべて CPU・ローカル推論で、公式配布の COCO 学習済み重みをそのまま使う（追加学習・しきい値の調整はしない）。
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv

Detector = Callable[[np.ndarray], sv.Detections]
NAMES = ("rfdetr", "yolox", "rtdetrv2", "dfine", "rfdetr-sahi", "yolox-sahi")

YOLOX_ONNX = Path("data/models/yolox_s.onnx")  # 公式リリース 0.1.1rc0 の yolox_s.onnx（README §9 で取得）
YOLOX_SIZE = 640
YOLOX_NMS_IOU = 0.45  # YOLOX の公式デモ（demo/ONNXRuntime）と同じ値
HF_MODELS = {"rtdetrv2": "PekingU/rtdetr_v2_r18vd", "dfine": "ustc-community/dfine-small-coco"}
SAHI_SLICE = 1080  # 縦（1080px）に合わせた正方形のタイル。モデル内の512への縮小が縦横とも約1/2で済む
SAHI_OVERLAP = 0.2


def load_detector(name: str, threshold: float) -> Detector:
    if name == "rfdetr":
        return _rfdetr(threshold)
    if name == "yolox":
        return _yolox(threshold)
    if name in HF_MODELS:
        return _hf(HF_MODELS[name], threshold)
    if name.endswith("-sahi"):
        return _sahi(load_detector(name.removesuffix("-sahi"), threshold), threshold)
    raise ValueError(f"未対応の検出器: {name}")


def _rfdetr(threshold: float) -> Detector:
    from rfdetr import RFDETRSmall
    from rfdetr.assets.coco_classes import COCO_CLASSES

    # COCO重みの predict が返す class_id は COCO_CLASSES のキー（0始まりの class_names とは別）
    person = [k for k, v in COCO_CLASSES.items() if v == "person"][0]
    model = RFDETRSmall(device="cpu")

    def detect(rgb: np.ndarray) -> sv.Detections:
        d = model.predict(rgb, threshold=threshold)
        return d[d.class_id == person]
    return detect


def _yolox(threshold: float) -> Detector:
    import onnxruntime as ort

    if not YOLOX_ONNX.is_file():
        raise FileNotFoundError(f"{YOLOX_ONNX} がありません（README §9 の手順で取得）")
    session = ort.InferenceSession(str(YOLOX_ONNX), providers=["CPUExecutionProvider"])
    # 出力は格子ごとの相対値なので、公式デモと同じく stride 8/16/32 の格子で画素へ戻す
    grids, strides = [], []
    for s in (8, 16, 32):
        n = YOLOX_SIZE // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    grid, stride = np.concatenate(grids), np.concatenate(strides)

    def detect(rgb: np.ndarray) -> sv.Detections:
        # 公式の前処理：BGR、縦横比を保って640へ縮小し、余白は114で埋める（正規化なし）
        h, w = rgb.shape[:2]
        r = min(YOLOX_SIZE / h, YOLOX_SIZE / w)
        resized = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w * r), int(h * r)),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((YOLOX_SIZE, YOLOX_SIZE, 3), 114, np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        out = session.run(None, {"images": padded.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy = (out[:, :2] + grid) * stride
        wh = np.exp(out[:, 2:4]) * stride
        score = out[:, 4] * out[:, 5]  # objectness × person（COCO の0番）
        keep = score >= threshold
        xy, wh, score = xy[keep], wh[keep], score[keep]
        boxes = np.c_[xy - wh / 2, xy + wh / 2] / r
        idx = cv2.dnn.NMSBoxes(np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]].tolist(),
                               score.tolist(), threshold, YOLOX_NMS_IOU)
        idx = np.array(idx, dtype=int).reshape(-1)
        return sv.Detections(xyxy=boxes[idx].astype(np.float32), confidence=score[idx].astype(np.float32),
                             class_id=np.zeros(len(idx), dtype=int))
    return detect


def _hf(repo: str, threshold: float) -> Detector:
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForObjectDetection.from_pretrained(repo).eval()
    person = [int(k) for k, v in model.config.id2label.items() if v == "person"][0]

    def detect(rgb: np.ndarray) -> sv.Detections:
        inputs = processor(images=rgb, return_tensors="pt")  # 前処理は配布元の設定どおり（640×640へ縮小）
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[rgb.shape[:2]])[0]
        d = sv.Detections(xyxy=res["boxes"].numpy().astype(np.float32),
                          confidence=res["scores"].numpy().astype(np.float32),
                          class_id=res["labels"].numpy().astype(int))
        return d[d.class_id == person]
    return detect


def _sahi(base: Detector, threshold: float) -> Detector:
    """任意の検出器を SAHI の分割推論で包む。

    横長のフレームを丸ごと縮小すると奥の選手が数ピクセルになるため、縦に合わせた正方形のタイルに分けて
    それぞれ検出し、フレーム全体の検出（SAHI 既定の perform_standard_pred）と合わせて重複をまとめる。
    """
    from sahi.models.base import DetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.prediction import ObjectPrediction

    class Wrapped(DetectionModel):
        def set_model(self, model, **kwargs):
            self.model = model
            self.category_mapping = {"0": "person"}

        def perform_inference(self, image, image_size=None):
            self._original_predictions = base(np.ascontiguousarray(image))

        def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=[[0, 0]],
                                                                     full_shape_list=None):
            # SAHI は [x, y] と [[x, y]] の両方の形で渡してくる（torchvision 版のラッパーと同じ扱い）
            if shift_amount_list and isinstance(shift_amount_list[0], (int, float)):
                shift_amount_list = [shift_amount_list]
            if full_shape_list and isinstance(full_shape_list[0], (int, float)):
                full_shape_list = [full_shape_list]
            shift = shift_amount_list[0] if shift_amount_list else [0, 0]
            full = full_shape_list[0] if full_shape_list else None
            d = self._original_predictions
            self._object_prediction_list_per_image = [[
                ObjectPrediction(bbox=box.tolist(), category_id=0, category_name="person", score=float(c),
                                 shift_amount=shift, full_shape=full)
                for box, c in zip(d.xyxy, d.confidence)]]

    model = Wrapped(model=base, confidence_threshold=threshold, device="cpu")

    def detect(rgb: np.ndarray) -> sv.Detections:
        res = get_sliced_prediction(rgb, model, slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                                    overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
                                    verbose=0)
        preds = res.object_prediction_list
        if not preds:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.array([p.bbox.to_xyxy() for p in preds], dtype=np.float32),
                             confidence=np.array([p.score.value for p in preds], dtype=np.float32),
                             class_id=np.zeros(len(preds), dtype=int))
    return detect
