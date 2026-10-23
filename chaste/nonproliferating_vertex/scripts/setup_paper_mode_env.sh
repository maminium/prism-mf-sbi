#!/usr/bin/env bash
# Source before paper-mode Yan quench Chaste runs.
# Usage: source scripts/setup_paper_mode_env.sh

export MFNNKB_YAN_MINIMAL_IO=1
# Full paper horizon unless you override (e.g. MFNNKB_YAN_END_TIME=8 for a quick smoke).
export MFNNKB_YAN_END_TIME="${MFNNKB_YAN_END_TIME:-700}"

# Chaste writes under build/testoutput unless overridden
export CHASTE_TEST_OUTPUT="${CHASTE_TEST_OUTPUT:-$HOME/chaste/build/testoutput}"

# Optional speed experiments (uncomment one):
# export MFNNKB_YAN_DISABLE_ADAPTIVE_DT=1

echo "Paper-mode Yan env set:"
echo "  MFNNKB_YAN_MINIMAL_IO=$MFNNKB_YAN_MINIMAL_IO"
echo "  MFNNKB_YAN_END_TIME=$MFNNKB_YAN_END_TIME"
echo "  CHASTE_TEST_OUTPUT=$CHASTE_TEST_OUTPUT"
echo "  CMAKE_BUILD_TYPE should be Release (see build/CMakeCache.txt)"
