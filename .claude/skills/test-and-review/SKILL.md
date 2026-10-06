---
name: test-and-review
description: sport-tracking-poc 固有版（global の test-and-review を上書きする）。track.py を短い実動画で段階実行し、出力（MP4・CSV）を確認して RESULT.md に記録する。「動作確認して」「実行して」「結果をまとめて」で使う。テスト基盤・lint は新設しない。
---

# PoCの動作確認と記録

テスト基盤やlint設定は作らない（ルート `CLAUDE.md`・`IMPLEMENTATION.md` の方針）。確認は実行と出力チェックで行う。

## 1. 環境チェック

```bash
.venv/bin/pip check
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # CPU版なら False
```

## 2. 段階実行（遅ければ途中で止めてよい）

1フレーム → 3秒 → 15秒の順。各回の**実行時間**（`time` の実出力）を控える。
15秒が遅すぎる場合は3〜5秒の結果で十分とし、改善ループは始めない。

## 3. 出力の確認

- MP4: 存在・サイズ・フレーム数（`cv2.VideoCapture` で `CAP_PROP_FRAME_COUNT`）。数フレームをPNGに書き出して目で見る。
- CSV: 行数、ユニークID数、列が仕様どおりか。
- 検出ゼロ・空CSVは「失敗/未確認」として扱い、成功と書かない。

## 4. RESULT.md

短く次を書く: 実行環境（CPU・Python・主要パッケージ版）／実行コマンド／各段の実行時間／出力の所在（Git外）／目視で分かったこと／問題点・限界。
**目視していない項目は「未確認」と明記**する。ID入れ替わり・見落としは記録するだけで直さない。
