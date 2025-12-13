#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LILITH_PERSONALITY_ENABLE=1
export LILITH_MOOD_ENABLE=1
export LILITH_PREFERENCES_ENABLE=1
export LILITH_DISABLE_MODAL=1
export LILITH_DISABLE_COMPOSITIONAL=1
export LILITH_DISABLE_PRAGMATIC=1
export LILITH_QUIET=${LILITH_QUIET:-1}
cd "$ROOT"
python lilith_cli.py "$@"
