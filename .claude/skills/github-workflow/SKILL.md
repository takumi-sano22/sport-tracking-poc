---
name: github-workflow
description: sport-tracking-poc 固有版（global の github-workflow を上書きする）。段階別 Issue（#2〜#5）ごとにブランチ→PR→自己レビュー＋reviewer agent→OKなら main へマージする軽量フロー。worktree・Codex は使わない。このリポジトリでコード・文書を変更するときに使う。
---

# PoC用の軽量 Git フロー

このリポジトリは使い捨てPoC（一次情報はルート `CLAUDE.md`）。2026-10-06 のユーザー決定により、Issue 単位で PR を出し、レビュー OK なら main へマージするまで自走する。

## Issue（作成済み・新規に増やさない）

| Issue | 内容 | ブランチ例 |
|---|---|---|
| #2 | 前提ゲート `preflight.sh` と運用ルール更新 | `chore/2-preflight-gate` |
| #3 | CPU環境構築と `track.py` 実装 | `feat/3-track` |
| #4 | 実動画での段階実行と `RESULT.md` | `docs/4-result` |
| #5 | 結果ビューア `viewer.html`（#4 の後） | `feat/5-viewer` |

## 手順

1. 着手前に `git status` を確認し、最新の main からブランチを切る（worktree は使わない）。
2. 小さな単位でコミットする。メッセージは日本語、prefix は `feat:` / `fix:` / `docs:` / `chore:`。
3. コミット前に `git diff --staged --stat` を見て、**素材・重み・出力動画・CSV・認証情報が入っていない**ことを確認する（`git add -A` より対象を明示した `git add`）。
4. ブランチを push し、`gh pr create` で PR を作る。本文に `Closes #N`、変更点、実行した確認の実出力を書く。
5. レビュー（2段）:
   - 自己レビュー: `git diff main...HEAD` を一読する。
   - `reviewer` agent: 差分をパッチファイルに保存して渡し、findings を P0〜P3 で返させる。
6. **OK の基準**: P0/P1 の指摘がゼロ。指摘があれば修正して再レビューする。2巡しても解消しなければマージせずユーザーに確認する。
7. OK なら `gh pr merge <番号> --merge --delete-branch` で main へマージし、`git switch main && git pull` で手元を更新する。
8. 完了報告では PR URL・マージコミットのハッシュ（実出力）を引用する。

## やらないこと

- main への直接 push、force push、履歴改変、既存ファイル削除。
- Codex レビュー、worktree、新しい Issue の追加（将来用 Issue は作らない）。
- 作業ログ: 別ファイルを作らない。結果と問題点は `RESULT.md` に書く。
