---
name: ui-design-recipes
description: sport-tracking-poc 固有版（global の ui-design-recipes を上書きする）。追跡結果（annotated.mp4・tracks.csv）を眺める簡易UIを作る・直すときに使う。道具としての見やすさだけを狙い、業務画面レシピ1本に絞る。
---

# PoC用のUIレシピ（最小）

このPoCのUIは**追跡結果を目で確かめるための道具**（動画の再生・ID一覧・簡単な集計程度）。
体験としての作り込みやデザイン体系の導入はしない（一次情報はルート `CLAUDE.md`）。

## 読むもの

- `references/layout/admin-dashboard.md` だけを読む。色は Tailwind 既定色の4系統、シェル・テーブル・空状態・グラフの型がそろっている。
- 文中の他の reference（`components/tooltip-and-hover-label.md` 等）はこのリポジトリに入れていない。必要になったときだけ global 版 `~/.claude/skills/ui-design-recipes/references/` を読む。

## PoCでの割り切り

- 1画面・静的ファイルで済ませる（ビルド基盤・サーバー・DB・認証を作らない）。CSVと動画はローカルから読む。
- 依存は増やさない。グラフが要れば自前の inline SVG。
- 生成物（動画・CSV）をUIのリポジトリ内へコピーしてコミットしない。
- 見た目の磨き込み・レスポンシブ対応の作り込みはしない。デスクトップで読めれば十分。
- 人物の役割や良し悪しを断定する表示（「上手い」「ミス」等）を出さない。仮IDは仮IDとして示す。
