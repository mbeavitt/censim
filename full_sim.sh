set -eu

mkdir -p output/fasta output/records output/generation_unit_pos

# Step 1: Initial Simulation (1000 generations)
python ./bin/simulate.py \
    ./data/15000copy_cen178.seq 1000 \
    ./data/15000copy_cen178.178bp.bed.pos \
    ./output/fasta/1000generation.out.fa \
    ./ouptut/records/1000generation.record.txt \
    ./output/generation_unit_pos/1000generation.unit.pos

## Step 2: Continue Simulation (to 6 million generations)
for i in $(seq 2000 1000 6000000); do
    prev=$((i - 1000))
    python ./bin/simulate.py \
        ./output/fasta/${prev}generation.out.fa 1000 \
        ./output/generation_unit_pos/${prev}generation.unit.pos \
        ./output/fasta/${i}generation.out.fa \
        ./output/records/${i}generation.record.txt \
       ./output/generation_unit_pos/${i}generation.unit.pos
done

for gen in $(seq 1000 1000 6000000); do
    echo ">centro_${gen}gen" | cat - ./output/fasta/${gen}generation.out.fa \
        > tmp && mv tmp ./output/fasta/${gen}generation.out.fa
done
