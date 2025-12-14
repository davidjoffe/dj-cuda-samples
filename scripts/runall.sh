#!/usr/bin/env bash
# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
# Helper to test run all built applications (say after build)

set -u

echo "dj: runall: Run all built applications"

# Colorize terminal output to look nicer and make it easier to see if all is working or if there are problems
# Define ANSI color codes
GREEN='\e[32m'
RED='\e[31m'
ORANGE='\e[33m' # Using yellow/orange for warnings
NC='\e[0m'     # No Color (Reset)

#BUILD_DIR="./build-linux/samples"
UNAME="$(uname)"

if [ "$UNAME" = "Darwin" ]; then
    BUILD_DIR="./build-macos/samples"
elif [ "$UNAME" = "Linux" ]; then
    BUILD_DIR="./build-linux/samples"
else
    echo "WARNING: UNKNOWN PLATFORM ($UNAME)"
    BUILD_DIR="./build-linux/samples"
fi

for dir in samples/*; do
    [ -d "$dir" ] || continue

    # e.g. "molecular_sim" => "./build-linux/samples/molecular_sim/djmolecular_sim"
    name=$(basename "$dir")
    bin="$BUILD_DIR/$name/dj${name}"

    echo "================================ $dir"
    if [ ! -f "$bin" ]; then
        #echo
        echo -e "dj: ${RED}WARNING: EXECUTABLE NOT FOUND:${NC}"
        echo -e "dj: ${ORANGE}$bin${NC}"
    else
        echo -e "dj: ${GREEN}Running $name${NC} ..."
        "$bin" "$@"
    fi
    #echo "================================"
    #echo
done
echo "dj: runall done"
