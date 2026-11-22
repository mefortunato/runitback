#!/usr/bin/env bash
# Publish a college football checkpoint as a GitHub release (tag cfb-VERSION) and open a PR that makes
# it the deployed model. Usage: publish_cfb_release.sh VERSION CHECKPOINT NOTES_FILE   (in GitHub
# Actions: needs GH_TOKEN with contents: write and pull-requests: write). The deploy workflow runs when
# that PR is merged into main.
set -euo pipefail
version="$1"; ckpt="$2"; notes="$3"
base="${RELEASE_BASE_BRANCH:-main}"
releases="src/runitback/cfb/releases.json"
tag="cfb-${version}"
asset="runitback-cfb-model-${version}.pt"

dir=$(mktemp -d)
cp "$ckpt" "$dir/$asset"
sha=$(sha256sum "$dir/$asset" | cut -d' ' -f1)
echo "publishing $asset ($sha)"

gh release create "$tag" "$dir/$asset" --title "College football model $version" --notes-file "$notes" \
  --target "${GITHUB_SHA:-$(git rev-parse HEAD)}" --latest=false

branch="release/cfb-${version}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git switch -c "$branch"
runitback cfb register-release "$version" --ckpt "$dir/$asset" --file "$releases"
git add "$releases"
git commit -q -m "Deploy college football model ${version}" -m "Release: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/releases/tag/${tag}
SHA-256: ${sha}"
git push -q origin "$branch"
{
  cat "$notes"
  printf '\n---\nMerging this PR makes `%s` the model baked into the Docker image (`cfb/releases.json`),\n' "$version"
  printf 'and the deploy workflow ships it. Close any older open college football release PRs.\n'
} > "$dir/body.md"
gh pr create --base "$base" --head "$branch" --title "Deploy college football model ${version}" --body-file "$dir/body.md"
