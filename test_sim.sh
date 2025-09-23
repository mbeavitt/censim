#!/bin/bash

set -eu

mkdir -p output/fasta output/records output/generation_unit_pos

GENERATIONS=${1:-1000}

# Test Simulation
py-spy record -o profile.svg -- python ./bin/simulate.py \
    ./data/15000copy_cen178.seq $GENERATIONS \
    ./data/15000copy_cen178.178bp.bed.pos \
    ./output/fasta/${GENERATIONS}generation.out.fa \
    ./output/records/${GENERATIONS}generation.record.txt \
    ./output/generation_unit_pos/${GENERATIONS}generation.unit.pos \
