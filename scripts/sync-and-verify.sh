#!/usr/bin/env bash
# Sync the local hermes-agent patch stack with upstream, verify, and back up to the fork.
#
# Usage: scripts/sync-and-verify.sh [--no-push]
#
# Flow: fetch origin -> rebase local main onto origin/main (rerere auto-resolves
# repeats) -> run the fork-patch regression suites + compat-pointer check ->
# push to the fork remote (force-with-lease; rebasing rewrites our patch SHAs,
# which is fine -- the fork is ours).
#
# On rebase conflict: fix files, `git add`, `git rebase --continue`, re-run.
# `git rebase --abort` always returns to the pre-sync state.
# See ~/=Hermes/Docs/Hermes/projects/skill-retrieval-rebuild/UPSTREAM-SYNC.md

set -euo pipefail
cd "$(dirname "$0")/.."

NO_PUSH=0
[ "${1:-}" = "--no-push" ] && NO_PUSH=1

echo "== fetch upstream =="
git fetch origin

echo "== rebase local main onto origin/main =="
if ! git rebase origin/main; then
  echo "" >&2
  echo "REBASE CONFLICT. Resolve the marked files, 'git add' them, then" >&2
  echo "'git rebase --continue'. (rerere remembers past resolutions and may" >&2
  echo "have pre-filled them.) Re-run this script afterwards. To bail out" >&2
  echo "cleanly: git rebase --abort" >&2
  exit 1
fi

echo "== regression: fork-patch suites =="
scripts/run_tests.sh \
  tests/agent/test_prompt_builder.py tests/agent/test_skill_utils.py \
  tests/agent/test_skill_commands.py tests/agent/test_external_skills.py \
  tests/agent/test_project_skills.py tests/agent/test_org_skill_namespace.py \
  tests/agent/test_skill_session_platform_gate.py \
  tests/tools/test_skills_tool.py tests/tools/test_skill_search_index.py \
  tests/tools/test_skills_tool_discovery_cache.py tests/tools/test_skill_usage.py \
  tests/tools/test_skill_manager_tool.py tests/test_cli_slash_suggest.py \
  tests/agent/test_display.py tests/agent/test_tool_guardrails.py \
  tests/agent/test_insights.py tests/agent/test_context_compressor.py

echo "== compat pointers =="
python3 scripts/check_compat_pointers.py

if [ "$NO_PUSH" = "1" ]; then
  echo "== done (push skipped) =="
  exit 0
fi

echo "== push to fork =="
git push --force-with-lease fork main

echo "== done: synced, verified, backed up =="
