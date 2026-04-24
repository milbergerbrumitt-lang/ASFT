#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/env.example.sh"
source "$ROOT_DIR/configs/smoke.env"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/tests:${PYTHONPATH:-}"

python "$ROOT_DIR/tests/run_smoke.py"
