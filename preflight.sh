#!/usr/bin/env bash
# 自走で実装に入る前の前提チェック（ゲート）。FAIL が1件でもあれば非0で終了する。
# 認証トークンは読み出さず・表示しない（状態だけを whoami の成否で見る）。
set -u
cd "$(dirname "$0")"

CLIP="data/mot/clips/118577.mp4"
PY=".venv/bin/python"
fail=0

pass() { echo "PASS  $1"; }
ng()   { echo "FAIL  $1"; fail=1; }
info() { echo "INFO  $1"; }

# --- Git / GitHub: PR 作成・マージに gh 認証が要る ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && pass "Gitリポジトリ" || ng "Gitリポジトリではない"
gh auth status >/dev/null 2>&1 && pass "gh 認証済み" || ng "gh 未認証（本人が gh auth login）"
info "ブランチ: $(git branch --show-current) / 未コミット変更: $(git status --porcelain | wc -l) 件"

# --- Python 仮想環境: 3.11〜3.12 を想定 ---
if [ -x "$PY" ]; then
  ver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  case "$ver" in 3.11|3.12) pass ".venv Python $ver" ;; *) ng ".venv Python $ver（3.11〜3.12 を想定）" ;; esac
else
  ng ".venv がない（python3 -m venv .venv）"
fi

# --- Hugging Face: 素材取得に本人の認証が要る。成否だけ見る ---
if [ -x .venv/bin/hf ] && .venv/bin/hf auth whoami >/dev/null 2>&1 \
   && ! .venv/bin/hf auth whoami 2>&1 | grep -qi "not logged in"; then
  pass "HF 認証済み"
else
  ng "HF 未認証（本人が .venv/bin/hf auth login）"
fi

# --- 素材: 1本だけ。サイズと MP4 ヘッダー（ftyp）で取り違え・途中終了を検出 ---
if [ -f "$CLIP" ]; then
  size=$(stat -c %s "$CLIP")
  if [ "$size" -gt 100000000 ] && head -c 12 "$CLIP" | grep -aq ftyp; then
    pass "素材 $CLIP ($((size / 1000000)) MB)"
  else
    ng "素材が不完全の可能性（$size bytes / ftyp なし）"
  fi
else
  ng "素材なし（hf download atomscott/soccertrack-v2 mot/clips/118577.mp4 --repo-type dataset --local-dir data）"
fi

# --- Git 除外: 素材・出力・重みがコミット対象にならないこと ---
ign_ok=1
for p in "$CLIP" outputs/x/annotated.mp4 outputs/x/tracks.csv weights/x.pth .env; do
  git check-ignore -q "$p" || { ng ".gitignore 対象外: $p"; ign_ok=0; }
done
[ "$ign_ok" -eq 1 ] && pass ".gitignore（素材・出力・重み・.env）"
tracked=$(git ls-files | grep -Ei '\.(mp4|csv|pth|pt|safetensors|onnx)$|(^|/)\.env' || true)
[ -z "$tracked" ] && pass "追跡中ファイルに素材・重み・認証情報なし" || ng "追跡中に除外すべきファイル: $tracked"

# --- 容量・ネットワーク: CPU wheel と重みの取得に必要 ---
avail=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
[ "$avail" -ge 10 ] && pass "ディスク空き ${avail}GB" || ng "ディスク空き ${avail}GB（10GB 以上を想定）"
for url in https://download.pytorch.org/whl/cpu/ https://pypi.org/simple/ https://huggingface.co/; do
  if [ -x "$PY" ] && "$PY" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=15)" "$url" >/dev/null 2>&1; then
    pass "到達可: $url"
  else
    ng "到達不可: $url"
  fi
done

# --- 参考情報（自走中に作るもの。FAIL にしない） ---
command -v ffmpeg >/dev/null 2>&1 && info "ffmpeg あり" || info "ffmpeg なし（必要時は .venv に imageio-ffmpeg を入れる）"
info "CPU $(nproc) コア / メモリ $(free -g | awk '/Mem:/ {print $2}')GB"

echo
[ "$fail" -eq 0 ] && echo "GATE: PASS" || echo "GATE: FAIL"
exit "$fail"
