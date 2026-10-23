#!/usr/bin/env bash
# Source before Fig. 2 / paper-mode Chaste runs (Step 1 profile).
# Usage: source scripts/setup_paper_mode_env.sh

export PRISM_VERTEX_PAPER_MODE=1
export PRISM_VERTEX_MINIMAL_IO=1
# Full paper horizon unless you override (e.g. PRISM_VERTEX_END_TIME=8 for a quick smoke).
export PRISM_VERTEX_END_TIME="${PRISM_VERTEX_END_TIME:-700}"

# Chaste writes under build/testoutput unless overridden
export CHASTE_TEST_OUTPUT="${CHASTE_TEST_OUTPUT:-$HOME/chaste/build/testoutput}"

# Optional speed experiments (uncomment one):
# export PRISM_VERTEX_DISABLE_ADAPTIVE_DT=1

echo "Paper-mode env set:"
echo "  PRISM_VERTEX_PAPER_MODE=$PRISM_VERTEX_PAPER_MODE"
echo "  PRISM_VERTEX_MINIMAL_IO=$PRISM_VERTEX_MINIMAL_IO"
echo "  PRISM_VERTEX_END_TIME=$PRISM_VERTEX_END_TIME"
echo "  CHASTE_TEST_OUTPUT=$CHASTE_TEST_OUTPUT"
echo "  CMAKE_BUILD_TYPE should be Release (see build/CMakeCache.txt)"
