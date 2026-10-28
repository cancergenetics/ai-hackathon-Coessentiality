#!/bin/sh
# Verify raw dataset checksums against DATA.md
cd "$(dirname "$0")/.." && grep -E '^\| (DepMAP|HuMAP|HGNC)/' DATA.md | awk -F'|' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $5"  datasets/"$2}' | shasum -a 256 -c
