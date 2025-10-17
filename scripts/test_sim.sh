#!/bin/bash

set -eu

mkdir -p output/fasta output/records output/generation_unit_pos output/cenh3

GENERATIONS=${1:-1000}

# Test Simulation
py-spy record -o profile.svg -- python scripts/test_sim.py $GENERATIONS
