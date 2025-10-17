#!/bin/bash

# Profile the optimized simulation with py-spy (10,000 generations)
echo "Profiling optimized simulation with py-spy (10,000 generations)..."

echo "Running first profile for JSON data..."
py-spy record --format speedscope -o profile_optimized.json -d 120 -- python scripts/profile_optimized.py

echo "Running second profile for SVG flamegraph..."
py-spy record -o profile_optimized.svg -d 120 -- python scripts/profile_optimized.py

echo "Profiling complete!"
echo "Flamegraph: profile_optimized.svg"
echo "JSON data: profile_optimized.json"
