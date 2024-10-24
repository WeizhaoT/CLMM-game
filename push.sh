#!/bin/bash
set -e

file="$1"
storage="$2"

mkdir -p "$file-$storage"
mv -f "$file/"*.{csv,jpg} "$file-$storage"
mv -f "$file/"*.json "$file-$storage" && rm -rf "$file"
