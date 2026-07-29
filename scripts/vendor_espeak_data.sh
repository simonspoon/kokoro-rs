#!/bin/sh
# Regenerate assets/espeak-ng-data.tar.gz.
#
# espeak-ng needs its data directory — compiled dictionaries, phoneme tables
# and voice definitions — at runtime. `espeak-rs-sys` builds espeak-ng from
# source and produces that directory inside its own OUT_DIR, but installs it
# nowhere useful, and cargo gives no ordering guarantee that it exists when
# this crate's build script runs. So the built directory is vendored here
# instead and embedded straight into the binary.
#
# Run this after bumping espeak-rs-sys, then commit the result.

set -eu

cd "$(dirname "$0")/.."

echo "building espeak-rs-sys..."
cargo build

data=$(find target -path '*/share/espeak-ng-data/phontab' -print 2>/dev/null | head -1)
if [ -z "$data" ]; then
    echo "error: no complete espeak-ng-data found under target/" >&2
    exit 1
fi
dir=$(dirname "$data")

echo "packing $dir"
mkdir -p assets
tar -czf assets/espeak-ng-data.tar.gz -C "$(dirname "$dir")" espeak-ng-data
ls -l assets/espeak-ng-data.tar.gz
