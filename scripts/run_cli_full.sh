#!/usr/bin/env bash
set -euo pipefail

# Full-featured Lilith CLI launcher with common dev/test switches enabled.
# Pass any additional CLI args through (e.g., "./run_cli_full.sh").

# Resolve repo root
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Toggle experiential layers
export LILITH_PERSONALITY_ENABLE=1
export LILITH_MOOD_ENABLE=1
export LILITH_PREFERENCES_ENABLE=1

# Observability
export LILITH_TRACE_SOURCES=1

# Cognitive / world grounding
export LILITH_ENABLE_WORLD_MODEL=1
# Uncomment to force-enable reasoning if you ever disable it elsewhere
# export LILITH_REASONING_ENABLE=1

# Relational concept retrieval + hygiene
export LILITH_ENABLE_RELATIONAL_CONCEPTS=1
export LILITH_CONCEPT_PROPERTY_MAX_LEN=160
export LILITH_CONCEPT_PROPERTY_REQUIRE_TERM=1

# Launch CLI
exec python3 "$ROOT_DIR/lilith_cli.py" "$@"
