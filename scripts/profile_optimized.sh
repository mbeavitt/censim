#!/bin/bash

# Profile the optimized simulation with py-spy (10,000 generations)
echo "Profiling optimized simulation with py-spy (10,000 generations)..."

echo "Running first profile for JSON data..."
py-spy record --format speedscope -o profile_optimized.json -d 120 -- python -m censim.simulation \
    ./data/15000copy_cen178.seq 10000 \
    ./data/15000copy_cen178.178bp.bed.pos \
    ./output/profile_test.fa \
    ./output/profile_test.record.txt \
    ./output/profile_test.pos \
    --seed 42

echo "Running second profile for SVG flamegraph..."
py-spy record -o profile_optimized.svg -d 120 -- python -m censim.simulation \
    ./data/15000copy_cen178.seq 10000 \
    ./data/15000copy_cen178.178bp.bed.pos \
    ./output/profile_test2.fa \
    ./output/profile_test2.record.txt \
    ./output/profile_test2.pos \
    --seed 42

echo "Profiling complete!"
echo "Flamegraph: profile_optimized.svg"
echo "JSON data: profile_optimized.json"
