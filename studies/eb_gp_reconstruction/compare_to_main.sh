#!/usr/bin/env bash
# Diff this tidied study tree against origin/main.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The tidy-up was done in a sandbox that cannot reach
# github.com/Liambcollins/ActiveModeMap: the repo is private, `gh` is not
# installed, and the GitHub API returns 403 for it.  The working copy on this
# machine also has no `.git` directory, so there was no local baseline either.
# Everything here was therefore built non-destructively -- renamed and moved,
# never deleted -- and this script is how you get the real comparison against
# main, from a clone that IS authenticated.
#
# USAGE
#   1. Clone (or fetch) the repo somewhere you can authenticate:
#          git clone https://github.com/Liambcollins/ActiveModeMap.git amm-main
#   2. Run this from the repo root of THIS working copy:
#          bash studies/eb_gp_reconstruction/compare_to_main.sh ../amm-main
#      or, if this working copy is itself a clone with a remote:
#          bash studies/eb_gp_reconstruction/compare_to_main.sh
#
# Writes a report to compare_to_main.report.txt and prints a summary.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"          # the ActiveModeMap package root
REPORT="$REPO/compare_to_main.report.txt"
OLD_NAME="physics_informed_al"
NEW_NAME="eb_gp_reconstruction"

# ---------------------------------------------------------------- locate main
REF=""
BASE=""
if [ $# -ge 1 ]; then
  BASE="$(cd "$1" && pwd)"
  echo "baseline: working tree at $BASE"
elif git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO" fetch origin main --quiet || {
    echo "could not fetch origin/main -- pass a path to a clone instead" >&2
    exit 1
  }
  REF="origin/main"
  echo "baseline: $REF"
else
  cat >&2 <<'MSG'
No .git here and no baseline path given.

  git clone https://github.com/Liambcollins/ActiveModeMap.git ../amm-main
  bash studies/eb_gp_reconstruction/compare_to_main.sh ../amm-main
MSG
  exit 1
fi

: > "$REPORT"
say () { echo "$*" | tee -a "$REPORT"; }
sec () { say ""; say "=== $* ==="; }

say "compare_to_main  --  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
say "this tree: $REPO"
say "baseline:  ${REF:-$BASE}"

# ------------------------------------------------------- file lists, old & new
NOISE='__pycache__|\.pyc$|\.report\.txt$'

list_base () {                                  # relative paths under studies/
  if [ -n "$REF" ]; then
    git -C "$REPO" ls-tree -r --name-only "$REF" -- "studies/" | sed 's|^studies/||'
  else
    ( cd "$BASE/ActiveModeMap/studies" 2>/dev/null || cd "$BASE/studies"
      find . -type f | sed 's|^\./||' )
  fi | grep -Ev "$NOISE"
}
list_new () {
  ( cd "$REPO/studies" && find . -type f | sed 's|^\./||' ) | grep -Ev "$NOISE"
}

list_base | sort > /tmp/cmp_base.txt
list_new  | sort > /tmp/cmp_new.txt

# strip the study-folder rename so real additions/removals stand out
sed "s|^$OLD_NAME/|<study>/|" /tmp/cmp_base.txt | sort > /tmp/cmp_base_n.txt
sed "s|^$NEW_NAME/|<study>/|" /tmp/cmp_new.txt  | sort > /tmp/cmp_new_n.txt

sec "study folder rename"
say "$OLD_NAME/  ->  $NEW_NAME/   ($(grep -c "^$OLD_NAME/" /tmp/cmp_base.txt) files on main)"

sec "files on main that are GONE from this tree (should be empty)"
comm -23 /tmp/cmp_base_n.txt /tmp/cmp_new_n.txt | tee -a "$REPORT" | sed -n '$='

sec "files NEW in this tree"
comm -13 /tmp/cmp_base_n.txt /tmp/cmp_new_n.txt | tee -a "$REPORT" >/dev/null
comm -13 /tmp/cmp_base_n.txt /tmp/cmp_new_n.txt

# ----------------------------------------------------------- content, per file
sec "files whose CONTENT changed (same logical path)"
while read -r rel; do
  old="${rel/<study>\//$OLD_NAME/}"
  new="${rel/<study>\//$NEW_NAME/}"
  case "$new" in *.png|*.pdf|*.pptx|*.npz|*.npy|*.zip|*.h5) bin=1;; *) bin=0;; esac
  if [ -n "$REF" ]; then
    a="$(git -C "$REPO" show "$REF:studies/$old" 2>/dev/null | md5sum | cut -d' ' -f1)"
  else
    src="$BASE/ActiveModeMap/studies/$old"; [ -f "$src" ] || src="$BASE/studies/$old"
    a="$(md5sum < "$src" 2>/dev/null | cut -d' ' -f1)"
  fi
  b="$(md5sum < "$REPO/studies/$new" 2>/dev/null | cut -d' ' -f1)"
  [ "$a" = "$b" ] && continue
  say "--- $rel"
  if [ "$bin" = 1 ]; then
    say "    (binary, md5 ${a:0:8} -> ${b:0:8})"
  elif [ -n "$REF" ]; then
    git -C "$REPO" diff --no-index --stat \
        <(git -C "$REPO" show "$REF:studies/$old") "$REPO/studies/$new" \
        2>/dev/null | tail -1 | tee -a "$REPORT"
  else
    diff -u "$src" "$REPO/studies/$new" | tail -n +3 | head -60 | tee -a "$REPORT"
  fi
done < <(comm -12 /tmp/cmp_base_n.txt /tmp/cmp_new_n.txt)

# -------------------------------------------------------------- sanity checks
sec "sanity"
# only .py matters: MIGRATION.md and this script mention those paths on purpose
LEFT="$(grep -rIl --include='*.py' '/home/claude\|/mnt/user-data' "$HERE" 2>/dev/null)"
say "scripts with container-absolute paths left: $(printf '%s' "$LEFT" | grep -c . || true)  (want 0)"
[ -n "$LEFT" ] && printf '%s\n' "$LEFT" | sed 's|^|    |' | tee -a "$REPORT"
say "scripts that byte-compile:"
( cd "$HERE" && python -m py_compile $(find . -name '*.py') >/dev/null 2>&1 \
  && say "  all OK" || say "  !! some failed -- run python -m py_compile yourself" )
say "quarantined, awaiting your review: $REPO/_to_delete  ($(du -sh "$REPO/_to_delete" 2>/dev/null | cut -f1))"

say ""
say "report written to $REPORT"
