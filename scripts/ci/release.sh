# Helpers for the Train / Refresh predictions workflows. `source` this file; each function takes
# the sport (or release tag) first, so a new sport's steps are calls, not copies of a loop.
#
# They tell "the release or asset isn't there yet" (a sport's first run) apart from "gh failed"
# (an outage, a rate limit): the first is handled, the second fails the step instead of training or
# continuing with nothing. Downloads go to dl/<sport>/, so two sports' `fold2025.tar.gz` never meet.

# release_assets <tag>: the release's asset names, one per line; nothing when it doesn't exist yet.
release_assets() {
  local out
  if out=$(gh release view "$1" --json assets -q '.assets[].name' 2>&1); then
    [ -z "$out" ] || printf '%s\n' "$out"
  elif grep -q "release not found" <<<"$out"; then
    return 0
  else
    echo "::error::cannot read release $1: $out" >&2
    return 1
  fi
}

# fetch_asset <tag> <asset> <dest dir>: download the asset into <dest dir>/ (replacing one that is
# there); fails when the release has no such asset.
fetch_asset() {
  local assets
  assets=$(release_assets "$1") || return 1
  grep -qx "$2" <<<"$assets" || { echo "::error::release $1 has no $2" >&2; return 1; }
  mkdir -p "$3" && gh release download "$1" --pattern "$2" --dir "$3" --clobber
}

# fetch_folds <sport> <root> <season>... [-- <season>...]: unpack <root>/fold<season>/ from the
# `foundation-<sport>` release. Every season before `--` must exist (the step fails otherwise); those
# after it are fetched when they do (a new season's fold, until Train has trained it).
fetch_folds() {
  local sport=$1 root=$2 rel="foundation-$1" assets s optional=0
  local -a seasons=() maybe=()
  shift 2
  for s in "$@"; do
    if [ "$s" = "--" ]; then optional=1
    elif [ $optional = 0 ]; then seasons+=("$s")
    else maybe+=("$s"); fi
  done
  assets=$(release_assets "$rel") || return 1
  for s in "${seasons[@]}"; do
    grep -qx "fold$s.tar.gz" <<<"$assets" || { echo "::error::$rel has no fold$s" >&2; return 1; }
  done
  for s in "${seasons[@]}" "${maybe[@]}"; do
    grep -qx "fold$s.tar.gz" <<<"$assets" || { echo "$rel: no fold$s yet"; continue; }
    gh release download "$rel" --pattern "fold$s.tar.gz" --dir "dl/$sport" --clobber || return 1
    mkdir -p "$root/fold$s" && tar -xzf "dl/$sport/fold$s.tar.gz" -C "$root/fold$s" || return 1
  done
}

# fetch_webdata [<sha256>]: download and unpack the `predictions-webdata` release's webdata.tar.gz
# into webdata/. With a sha256 (the asset Train predictions just uploaded, or the one an earlier job
# of the same run fetched), wait for that asset: the release can still hold the previous one.
fetch_webdata() {
  local want=${1:-} got attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    rm -f webdata.tar.gz
    gh release download predictions-webdata --pattern webdata.tar.gz || return 1
    got="$(sha256sum webdata.tar.gz | cut -d' ' -f1)"
    if [ -z "$want" ] || [ "$got" = "$want" ]; then break; fi
    if [ "$attempt" = 10 ]; then
      echo "::error::predictions-webdata asset is $got, not $want" >&2; return 1
    fi
    echo "predictions-webdata asset is still the previous one ($got); retrying"
    sleep 30
  done
  rm -rf webdata && mkdir webdata && tar -xzf webdata.tar.gz -C webdata
}
