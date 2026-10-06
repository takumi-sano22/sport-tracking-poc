# sport-tracking-poc

**サッカー動画に人物の枠・仮のID・短い軌跡を付けるだけの、使い捨てPoC。**

本格版は別リポジトリで作る。このリポジトリでは、拡張性や精度向上を追わず、最小構成の動きと限界を一度見る。

> `track.py`は実装済み（#3）。実動画での実行結果は`RESULT.md`に記録する。
> 結果の一次記録は`RESULT.md`、外部向けのまとめ（素材・処理・所感）は`REPORT.md`。

## 決定済み

| 項目 | 今回の方針 |
|---|---|
| 配置 | `takumi-sano22/sport-tracking-poc` / public |
| 実行 | 手元のWSL / Linux、CPU、Python仮想環境 |
| 素材 | SoccerTrack v2のMOT用クリップ1本。Hugging Faceへのログイン・条件同意は本人が行う |
| 初期構成 | RF-DETR Small → `trackers`のByteTrack → Supervision / OpenCVで描画・保存 |
| 対象 | 汎用の`person`検出。選手と審判・スタッフの区別はしない |
| 処理区間 | まず1フレーム、次に3秒、動いたら15秒。CPUで重ければ3〜5秒の結果で終えてよい |
| 結果 | ID・軌跡付きMP4、画面座標のCSV、短い`RESULT.md` |
| 任意 | 出力（MP4・CSV）を眺める結果ビューア。静的HTML 1画面のみ |
| ピッチ座標 | #11で追加。配布元のキャリブレーションで足元を俯瞰xy（m）へ変換し、正解と比較（§8） |
| 対象外 | 学習、Re-ID、背番号・氏名、ボール、チーム分類、複数カメラ、サーバー/API/DB、Docker、CI/CD、クラウド、ベンチマーク |
| Claude Code設定 | `CLAUDE.md`と`.claude/`（PoC用に縮小したskill・権限設定）。詳細は`CLAUDE.md` |

「IDが付いた」と「正しく同じ人物を追えた」は別。IDの入れ替わり・見落としも結果として記録する。

## 1. リポジトリをWSLへcloneする

このリポジトリはすでに作成済み。WSLの任意の作業場所へcloneする。

```bash
mkdir -p ~/project
cd ~/project
git clone https://github.com/takumi-sano22/sport-tracking-poc.git
cd sport-tracking-poc
```

GitHub CLIを使う場合は次でもよい。

```bash
gh repo clone takumi-sano22/sport-tracking-poc ~/project/sport-tracking-poc
cd ~/project/sport-tracking-poc
```

## 2. Claude Codeへ渡す

このフォルダでClaude Codeを開き、`START_CLAUDE.txt`を渡す。
最初に`CLAUDE.md`と`IMPLEMENTATION.md`を読み、実装から短い実動画の実行まで進めてもらう。
素材への同意・認証が未完了でも、コードの作成や入出力のテストは先に進めてよい。

## 3. 素材の同意・ログイン（本人の操作）

[SoccerTrack v2の配布ページ][s1]へブラウザでログインし、表示される利用条件・連絡先共有を確認して同意する。
認証トークンをClaude Codeの会話やGitHubへ貼らない。

Claude Codeが作った`.venv`を使う。まだなければ、次のように作れる。

```bash
python3 -m venv .venv  # 未作成の場合だけ
source .venv/bin/activate
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

`hf auth login`の案内に従って本人が認証する。トークン入力方式なら読み取り用を使い、
Git認証情報への登録は不要。ブラウザでのデータセット同意とCLIの認証は別の操作。[s3]

## 4. 取得するのは1本だけ

対象は`mot/clips/118577.mp4`。公式一覧の表示は約270 MB。[s2]
データセット全体をclone / downloadしない。正解アノテーションも今回の実行には不要。
素材を取得するのは条件同意・認証後とする。

```bash
mkdir -p data
hf download atomscott/soccertrack-v2 mot/clips/118577.mp4 \
  --repo-type dataset --local-dir data
```

保存先：`data/mot/clips/118577.mp4`。取得日は`RESULT.md`へ記録する。
権限エラー時は本人の同意・ログイン状態を確認し、別のミラーで回避しない。
ファイルの配布構成が変わった場合は公式一覧を再確認し、MOTクリップ1本だけを選び直して記録する。

## 5. CPU環境の構築

Python 3.12の`.venv`で、2026-10-06に次の版で動作を確認した（全体は`requirements.txt`）。

| パッケージ | 版 |
|---|---|
| torch / torchvision | 2.14.1+cpu / 0.29.1+cpu |
| rfdetr | 1.11.2（重み `rf-detr-small.pth`、COCO学習済み） |
| trackers | 2.6.1（`ByteTrackTracker`） |
| supervision | 0.30.7 |
| opencv-python | 5.0.0.93 |

```bash
python3 -m venv .venv  # 未作成の場合だけ
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"  # False であること
```

- `+cpu`版のtorch / torchvisionはPyPIに無く、`requirements.txt`先頭の`--extra-index-url https://download.pytorch.org/whl/cpu`から取得する。
- OpenCVは`trackers`が`opencv-python`（GUI版）を必須にしているため、headless版を入れず GUI版1つにしている（GUI機能は使わない）。
- 重みは初回実行時に`rfdetr`が`~/.roboflow/models/rf-detr-small.pth`（約368 MB）へ自動取得する。リポジトリには入れない。
- `transformers`の依存解決で`huggingface_hub`は1.33.0になる（HF認証はそのまま使えることを`preflight.sh`で確認済み）。

## 6. 実装後の実行例

```bash
source .venv/bin/activate

# 動作確認。元クリップの先頭から3秒間、フレームを間引かず処理する。
python track.py --input data/mot/clips/118577.mp4 \
  --start 0 --duration 3 --output outputs/smoke

# 動いたら15秒。遅すぎる場合は3〜5秒で止め、実行時間を記録してよい。
python track.py --input data/mot/clips/118577.mp4 \
  --start 0 --duration 15 --output outputs/demo
```

`outputs/<実行名>/annotated.mp4`と`tracks.csv`を確認する。
音声は不要。動画再生にWSLのGUIは必須にせず、保存したMP4をWindows側で開けばよい。
`RESULT.md`には再実行コマンド、実行時間、観察した問題を短く残す。

## 7. 結果ビューア（viewer.html）

`track.py`の出力はOpenCVの`mp4v`形式で、ブラウザでは再生できないことがある。
`.venv`の`imageio-ffmpeg`（0.6.0、同梱FFmpeg 7.0.2）でH.264へ変換する。sudoやシステムのFFmpegは使わない。

```bash
source .venv/bin/activate
python -m pip install imageio-ffmpeg==0.6.0  # requirements.txt に含まれる
FF=$(python -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())")
"$FF" -loglevel error -i outputs/demo/annotated.mp4 -c:v libx264 -pix_fmt yuv420p \
  -crf 23 -preset veryfast -movflags +faststart -an outputs/demo/annotated_h264.mp4
```

`viewer.html`をブラウザで直接開き（サーバー不要）、`annotated_h264.mp4`と`tracks.csv`をファイル選択で読み込む。
表示するもの：動画、集計（CSV行数・仮ID数など）、表示中のフレームのID、ID一覧（行をクリックすると、そのIDの最初のフレームへ移動）。
`--start`を0以外で実行した場合は、画面の「開始秒」を合わせる。変換は画質を落とす再エンコードで、内容（枠・ID）は変えない。

## 8. ピッチ座標（俯瞰xy）への変換（#11）

`track.py`の`tracks.csv`の足元の点（枠の下辺の中央）を、ピッチを上から見た座標（m）へ変換する。
座標系はSoccerTrack v2の規約に合わせる：105m×68m、原点はセンターサークル、xはメインカメラから見て右向き、yはカメラ側のタッチライン向きが正。

変換：足元の画素 → 魚眼の歪み補正（配布元のK・D）→ 配布元のホモグラフィの逆行列 → 手作業で付けられた65点のキーポイントとの残差を薄板スプライン（TPS）で補正。

### 追加の素材（本人の認証済みの環境で取得）

```bash
source .venv/bin/activate
# キャリブレーション（小さい。必須）
hf download atomscott/soccertrack-v2 \
  raw/118577/118577_camera_intrinsics.npz raw/118577/118577_homography.npy raw/118577/118577_keypoints.json \
  --repo-type dataset --local-dir data
# 評価用の正解（eval_pitch.py を使う場合だけ。mot は 6MB、gsr の前半は約2.9GB）
hf download atomscott/soccertrack-v2 mot/118577.txt gsr/118577/118577_1st.json --repo-type dataset --local-dir data
```

`camera_intrinsics.npz`のrvecs/tvecsはpickle形式のため読まない（`allow_pickle`を使わない）。

### 実行

```bash
python pitch.py --tracks outputs/demo/tracks.csv --calib data/raw/118577 \
  --video data/mot/clips/118577.mp4 --start 0 --output outputs/demo_pitch
python eval_pitch.py --gsr data/gsr/118577/118577_1st.json --mot data/mot/118577.txt \
  --calib data/raw/118577 --pitch-tracks outputs/demo_pitch/pitch_tracks.csv --output outputs/demo_eval
```

| 出力 | 内容 |
|---|---|
| `pitch_tracks.csv` | frame_index, timestamp_sec, track_id, x_m, y_m, in_pitch（ラインから外側2m以内なら1）, confidence |
| `pitch_summary.csv` | ID別の出現フレーム数・走行距離・平均速度・1秒間の最高速度（IDが付け直されるため参考値） |
| `topview.mp4` | 俯瞰図（H.264）。ピッチ外は灰色。`viewer.html`の「俯瞰動画」で元動画と同期表示できる |
| `calib_overlay.png` / `calibration_check.json` | ピッチのラインの重ね描きと、キーポイントの検証誤差 |
| `pitch_eval.json`（eval_pitch.py） | クリップの区間の特定結果と、正解との誤差 |

`eval_pitch.py`は初回にgsrから必要な値だけを抜き出し、`data/gsr/.../*.extract.npz`へキャッシュする（約50秒、メモリ約0.5GB）。

## ソースと利用条件

確認日：2026-10-06。下記は配布元・公式資料で確認した範囲であり、実行検証済みという意味ではない。
実装時に使ったパッケージ版・重み名を固定して記録する。

| 対象 | 確認した内容 |
|---|---|
| SoccerTrack v2 [s1][s2] | データはCC BY 4.0表示。取得にはログイン・条件同意が必要。MOT用の短い動画を配布 |
| RF-DETR [s4] | オープンソースコードとApache指定モデルを使う。Smallを対象とし、Plusや別条件の重みは使わない |
| RF-DETR Small API [s5] | 検出出力は`supervision.Detections`。RGB入力とクラス名の対応を確認する |
| Roboflow `trackers` [s6] | Apache-2.0のByteTrack再実装。`ByteTrackTracker.update(detections)`で接続 |
| Supervision [s7] | MIT。描画・検出データの共通表現に使用 |
| HF CLI [s3] / GitHub CLI [s8] / PyTorch [s9] | 取得・認証、新規リポジトリ作成、CPU実行環境の公式手順 |

素材へのクレジット：SoccerTrack v2 — Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim, Keisuke Fujii。
データセット：[配布元][s1]、論文：[SoccerTrack v2][s10]、ライセンス：[CC BY 4.0][s11]。
PoC出力では区間抽出、人物枠・ID・軌跡の重畳を行う。再生用に形式を変えた場合はその変更も記録する。
素材・重み・出力動画はGitへ入れない。出力を別途共有する際は、出典・ライセンス・変更内容も添える。

[s1]: https://huggingface.co/datasets/atomscott/soccertrack-v2
[s2]: https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main/mot/clips
[s3]: https://huggingface.co/docs/huggingface_hub/guides/cli
[s4]: https://github.com/roboflow/rf-detr
[s5]: https://rfdetr.roboflow.com/reference/small/
[s6]: https://github.com/roboflow/trackers
[s7]: https://github.com/roboflow/supervision
[s8]: https://cli.github.com/manual/gh_repo_create
[s9]: https://pytorch.org/get-started/locally/
[s10]: https://arxiv.org/abs/2508.01802
[s11]: https://creativecommons.org/licenses/by/4.0/
