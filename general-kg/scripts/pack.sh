#!/bin/bash
set -eo pipefail

echo "[pack] locking dependencies"
uv lock
