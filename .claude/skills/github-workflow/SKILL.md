---
name: github-workflow
description: sport-tracking-poc 固有版（global の github-workflow を上書きする）。使い捨てPoCのため Issue / PR / worktree / 2段レビューを使わず、ローカルコミットまでで完結させる軽量フロー。このリポジトリでコード・文書を変更するときに使う。
---

# PoC用の軽量 Git フロー

このリポジトリは1日で終える使い捨てPoC（一次情報はルート `CLAUDE.md`）。
global 版のフロー（Issue → worktree → PR → 自己レビュー → 第2段レビュー）は規模に見合わないので**使わない**。

## 手順

1. 着手前に `git status` で作業ツリーを確認する。
2. main 上で直接作業してよい（worktree・Issue は作らない）。
3. 変更は小さな単位でローカルコミットする。メッセージは日本語、prefix は `feat:` / `fix:` / `docs:` / `chore:`。
4. コミット前に `git diff --staged --stat` を見て、**素材・重み・出力動画・CSV・認証情報が入っていない**ことを確認する（`.gitignore` 頼みにしない。`git add -A` より対象を明示した `git add` を使う）。
5. 完了報告ではコミットハッシュ（`git log --oneline -1` の実出力）を引用する。

## やらないこと

- `git push` / PR 作成 / Issue 作成: ユーザーの明示指示があるときだけ。
- レビュー: subagent・Codex による第2段レビューは行わない。自分で差分を一読すれば十分。
- 作業ログ: 別ファイルを作らない。結果と問題点は `RESULT.md` に書く。
- force push・履歴改変・既存ファイル削除。
