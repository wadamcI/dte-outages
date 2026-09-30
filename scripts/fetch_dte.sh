#!/usr/bin/env bash
# Fetch every DTE outage polygon (paginated ArcGIS query) and write a
# normalized FeatureCollection to $OUT_DIR/outages.geojson.
#
# Writes "changed=true|false" to $GITHUB_OUTPUT when running in Actions, and a
# commit message to $OUT_DIR/../dte-message.txt when the snapshot changed.
set -euo pipefail

BASE='https://outagemap.serv.dteenergy.com/GISRest/services/OMP/OutageLocations/MapServer/2/query'
COMMON='text=%25&outFields=%2A&returnGeometry=true&outSR=4326&f=geojson&orderByFields=OBJECTID%20ASC'
PAGE_SIZE=1000
MAX_PAGES=50

OUT_DIR="${OUT_DIR:-dte}"
CONTACT_EMAIL="${CONTACT_EMAIL:-wadamc@umich.edu}"
REPO_URL="${REPO_URL:-https://github.com/wadamcI/dte-outages}"
# Identify ourselves to DTE so their operators know who is polling and how to
# reach us. "From" is the standard HTTP header for a responsible contact.
USER_AGENT="dte-outages-archiver/2.0 (+${REPO_URL}; contact: ${CONTACT_EMAIL})"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
warned_tls=false

fetch() { # fetch <url> <dest>
  local args=(-sS --fail --compressed --max-time 60 --retry 3 --retry-delay 10
              -A "$USER_AGENT" -H "From: ${CONTACT_EMAIL}" -o "$2")
  local rc=0
  curl "${args[@]}" "$1" || rc=$?
  # 60 = peer certificate cannot be authenticated (e.g. expired certificate).
  if [[ $rc -eq 60 && "${ALLOW_INSECURE_TLS_FALLBACK:-false}" == "true" ]]; then
    if [[ $warned_tls == false ]]; then
      echo "::warning::TLS verification failed for ${1%%\?*} (certificate invalid/expired); retrying without verification because ALLOW_INSECURE_TLS_FALLBACK=true."
      warned_tls=true
    fi
    rc=0
    curl "${args[@]}" --insecure "$1" || rc=$?
  fi
  return $rc
}

mkdir -p "$OUT_DIR"
offset=0
pages=()
for ((i = 0; i < MAX_PAGES; i++)); do
  page="$tmp/page-$offset.json"
  echo "Fetching DTE offset=$offset"
  fetch "$BASE?$COMMON&resultRecordCount=$PAGE_SIZE&resultOffset=$offset" "$page"

  # ArcGIS returns HTTP 200 with {"error": {...}} on query failures.
  if ! jq -e '.type == "FeatureCollection" and (.features | type == "array")' "$page" >/dev/null 2>&1; then
    echo "::error::Unexpected response at offset $offset: $(head -c 300 "$page")"
    exit 1
  fi

  pages+=("$page")
  count=$(jq '.features | length' "$page")
  more=$(jq '(.exceededTransferLimit // .properties.exceededTransferLimit // false)' "$page")
  if (( count == 0 )) || { (( count < PAGE_SIZE )) && [[ $more != true ]]; }; then
    break
  fi
  offset=$((offset + count))
done

jq -s -S '{type: "FeatureCollection",
           features: (map(.features[]) | unique_by(.properties.OBJECTID) | sort_by(.properties.OBJECTID))}' \
  "${pages[@]}" > "$tmp/new.geojson"

old="$OUT_DIR/outages.geojson"
new_count=$(jq '.features | length' "$tmp/new.geojson")
old_count=0
changed=true
if [[ -f $old ]]; then
  old_count=$(jq '.features | length' "$old")
  jq -S '.features |= sort_by(.properties.OBJECTID)' "$old" > "$tmp/old.geojson"
  if cmp -s "$tmp/old.geojson" "$tmp/new.geojson"; then
    changed=false
  fi
fi

if [[ $changed == true ]]; then
  mv "$tmp/new.geojson" "$old"
  printf 'DTE outage update: %s features=%s (prev=%s)\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$new_count" "$old_count" > "$OUT_DIR/../dte-message.txt"
  echo "DTE: $new_count features (prev=$old_count)."
else
  echo "DTE: no changes ($new_count features)."
fi

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  echo "changed=$changed" >> "$GITHUB_OUTPUT"
fi
