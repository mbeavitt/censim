#!/bin/bash

set -eu

# Test Simulation (10000 generations)
py-spy record -o profile.svg -- python ./bin/simulate.py \
    ./data/15000copy_cen178.seq 10000 \
    ./data/15000copy_cen178.178bp.bed.pos \
    ./output/fasta/1000generation.out.fa \
    ./output/records/1000generation.record.txt \
    ./output/generation_unit_pos/1000generation.unit.pos

