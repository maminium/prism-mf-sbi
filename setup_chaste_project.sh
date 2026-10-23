#!/usr/bin/env bash
# Symlink this repo's Chaste user projects into a local Chaste tree.
#
# Usage:  ./setup_chaste_project.sh <CHASTE_SRC>
#         (defaults to ~/chaste if <CHASTE_SRC> is not given)
#
# This repository is NOT a Chaste install. Install Chaste separately, then run
# this script so the simulators appear under <CHASTE_SRC>/projects/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CHASTE_SRC="${1:-$HOME/chaste}"

if [ ! -d "${CHASTE_SRC}" ]; then
    echo "Error: ${CHASTE_SRC} is not a directory."
    echo "Install Chaste first (https://chaste.github.io/docs/installguides/)."
    exit 1
fi

mkdir -p "${CHASTE_SRC}/projects"

link_project() {
    local src="$1"
    local name="$2"
    local dst="${CHASTE_SRC}/projects/${name}"
    if [ ! -d "${src}" ]; then
        echo "Skip ${name}: missing ${src}"
        return 0
    fi
    if [ -e "${dst}" ] || [ -L "${dst}" ]; then
        echo "Removing existing ${dst}"
        rm -rf "${dst}"
    fi
    ln -s "${src}" "${dst}"
    echo "Linked ${src} -> ${dst}"
}

link_project "${REPO_ROOT}/chaste/proliferating_vertex" "ProliferatingVertex"
link_project "${REPO_ROOT}/chaste/nonproliferating_vertex" "NonproliferatingVertex"

echo
echo "Next (proliferating):"
echo "  cd ${CHASTE_SRC}/build"
echo "  cmake -DCMAKE_BUILD_TYPE=Release ${CHASTE_SRC}"
echo "  make -j project_ProliferatingVertex TestPaperCommandLineSpeedSimulation"
echo
echo "Next (non-proliferating, if present):"
echo "  make -j project_NonproliferatingVertex TestYanVertexCommandLine"
echo
echo "See the repository README for environment variables."
