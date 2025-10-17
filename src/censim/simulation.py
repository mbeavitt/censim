import random
import numpy as np
import argparse
import re
import pandas as pd



def read_sequence(file_name):
    with open(file_name, "r") as file:
        sequence = file.read().strip()
    return sequence

def read_pos_file(file_path):
    """Read and parse a BED file, returning a DataFrame containing start positions with integer type."""
    return pd.read_csv(file_path, sep="\t", header=None, usecols=[1], names=["start"], dtype={"start": int})

def adjust_pos_coordinates(pos1, pos2):
    """Adjust pos1 coordinates based on pos2 instructions (INS/DEL)."""
    if not pos2:
        return pos1

    # Convert to numpy array for vectorized operations
    positions = np.array(pos1, dtype=np.int32)

    for _, ins_del, p1, p2 in pos2:
        if ins_del == "DEL":
            # Create boolean masks for each condition
            mask_lt_p1 = positions < p1
            mask_between = (positions >= p1) & (positions < p2)
            mask_gte_p2 = positions >= p2

            # Apply transformations using masks
            # Positions < p1: no change
            # Positions between p1 and p2: remove (handled by not including them)
            # Positions >= p2: subtract (p2 - p1)
            new_positions = positions[mask_lt_p1 | mask_gte_p2].copy()
            new_positions[positions[mask_lt_p1 | mask_gte_p2] >= p2] -= (p2 - p1)
            positions = new_positions

        elif ins_del == "INS":
            # Create boolean masks
            mask_lt_p1 = positions < p1
            mask_between = (positions >= p1) & (positions < p2)
            mask_gte_p2 = positions >= p2

            # Calculate new positions
            pos_lt_p1 = positions[mask_lt_p1]  # no change
            pos_between = positions[mask_between]  # duplicate with offset
            pos_gte_p2 = positions[mask_gte_p2] + (p2 - p1)  # shift by insertion size

            # Concatenate: original + duplicated + shifted
            positions = np.concatenate([
                pos_lt_p1,
                pos_between,
                pos_between + (p2 - p1),
                pos_gte_p2
            ])

    return positions.tolist()

def apply_snp_mutations(seq, generation, records):
    """Apply SNP mutations to sequence."""
    count = 0
    target = np.random.poisson(0.1)
    bases = ['A', 'T', 'C', 'G']

    while count < target:
        idx = random.randint(0, len(seq) - 1)
        available = [b for b in bases if b != seq[idx]]
        new_base = random.choice(available)

        records.append((generation, "SNP", idx + 1, seq[idx], new_base, "-"))
        seq[idx] = new_base
        count += 1

def apply_indel_mutations(seq, generation, records, indel_records, repeat_size=178, max_retries=5000):
    """Apply INDEL mutations to sequence using whole repeats with modulo arithmetic.

    Args:
        seq: Sequence as list
        generation: Current generation number
        records: Mutation records list
        indel_records: INDEL coordinate adjustment records
        repeat_size: Size of each repeat unit in bp (default: 178)
        max_retries: Maximum number of consecutive failures before signaling collapse (default: 5000)

    Returns:
        bool: True if array collapsed, False otherwise
    """
    count = 0
    target = np.random.poisson(0.5)
    consecutive_failures = 0

    while count < target:
        # Recalculate number of units based on current sequence length
        num_units = len(seq) // repeat_size

        # Pick a random unit number
        unit_num = random.randint(0, num_units - 1)
        indel_type = random.choice(["INS", "DEL"])
        copy_num = np.random.poisson(7.6)

        # Skip if copy_num is 0 or negative
        if copy_num <= 0:
            continue

        # Calculate target unit (copy_num repeats away)
        target_unit_num = unit_num + copy_num

        # Bounds check: ensure target unit exists
        if target_unit_num >= num_units:
            consecutive_failures += 1
            if consecutive_failures >= max_retries:
                return True  # Signal array collapse
            continue

        # Reset failure counter on success
        consecutive_failures = 0

        # Calculate boundaries using modulo arithmetic (whole repeats only)
        unit_start = unit_num * repeat_size
        target_start = target_unit_num * repeat_size

        # Extract the sequence between the two repeat boundaries
        indel_seq = "".join(seq[unit_start:target_start])

        if indel_type == "INS":
            # Insert the sequence at unit_start
            records.append((generation, indel_type, unit_start, seq[unit_start - 1] if unit_start > 0 else '',
                          seq[unit_start - 1] if unit_start > 0 else '' + indel_seq, copy_num))
            seq[unit_start:unit_start] = list(indel_seq)
        else:  # DEL
            # Delete the sequence from unit_start to target_start
            records.append((generation, indel_type, unit_start, indel_seq,
                          seq[unit_start - 1] if unit_start > 0 else '', copy_num))
            del seq[unit_start:target_start]

        indel_records.append((generation, indel_type, unit_start, target_start))
        count += 1

    return False  # No collapse occurred



def update_cenh3(cenh3_occupancy, indel_records):
    """Update CENH3 occupancy array based on INDEL records. Currently a placeholder."""
    return cenh3_occupancy


def initialize_cenh3_occupancy(num_units):
    """Initialize CENH3 occupancy with normal distribution, rate 0.4 at the center (mode).

    Overall occupancy rate is approximately 2-3% with peak concentration at center.
    """
    cenh3_occupancy = np.zeros(num_units, dtype=bool)
    middle = num_units // 2
    std_dev = num_units / 6

    # For each unit, calculate probability based on normal distribution
    # At center (distance=0), probability = 0.4 (the mode/peak)
    for i in range(num_units):
        distance = abs(i - middle)
        prob = 0.4 * np.exp(-(distance**2) / (2 * std_dev**2))
        cenh3_occupancy[i] = np.random.random() < prob

    return cenh3_occupancy

def introduce_mutations(sequence, generation, num_generations, repeat_size=178):
    """Introduce mutations into sequence over multiple generations.

    Args:
        sequence: Initial DNA sequence string
        generation: Starting generation number
        num_generations: Number of generations to simulate
        repeat_size: Size of each repeat unit in bp (default: 178)

    Returns:
        tuple: (mutated_sequence, mutation_records, cenh3_occupancy, collapsed)
            where collapsed is True if the array collapsed to zero, False otherwise
    """
    seq = list(sequence)
    records = []

    # Initialize CENH3 occupancy based on initial sequence length
    num_units = len(sequence) // repeat_size
    cenh3_occupancy = initialize_cenh3_occupancy(num_units)

    collapsed = False
    for _ in range(num_generations):
        generation += 1

        apply_snp_mutations(seq, generation, records)

        # Check if array collapsed during INDEL mutations
        collapsed = apply_indel_mutations(seq, generation, records, [], repeat_size)
        if collapsed:
            print("Simulation complete: Array collapsed to zero")
            break

    return "".join(seq), records, cenh3_occupancy, collapsed

