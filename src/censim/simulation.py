import random
import numpy as np
import argparse
import re
import pandas as pd
import bisect
from Bio import Align

# Global aligner instance to avoid recreation overhead
_global_aligner = None

def get_aligner():
    """Get or create the global pairwise aligner instance."""
    global _global_aligner
    if _global_aligner is None:
        _global_aligner = Align.PairwiseAligner()
        _global_aligner.match_score = 2
        _global_aligner.mismatch_score = -1
        _global_aligner.open_gap_score = -10
        _global_aligner.extend_gap_score = -1
    return _global_aligner

def find_unit_boundaries(unit_positions, idx):
    """Fast unit boundary lookup using binary search."""
    if len(unit_positions) == 0:
        return None, None

    # Find the last unit start <= idx
    pos = bisect.bisect_right(unit_positions, idx) - 1
    if pos < 0:
        return None, None

    unit_start = unit_positions[pos]

    # Find the next unit start > idx
    if pos + 1 < len(unit_positions):
        unit_end = unit_positions[pos + 1]
    else:
        return unit_start, None

    return unit_start, unit_end

def find_nth_unit_after(unit_positions, idx, n):
    """Find the nth unit after the given index."""
    if len(unit_positions) == 0 or n <= 0:
        return None, None

    # Find first unit start > idx
    pos = bisect.bisect_right(unit_positions, idx)

    if pos + n - 1 >= len(unit_positions):
        return None, None

    start_pos = pos + n - 1
    unit_start = unit_positions[start_pos]

    if start_pos + 1 < len(unit_positions):
        unit_end = unit_positions[start_pos + 1]
    else:
        return unit_start, None

    return unit_start, unit_end

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

def get_sequence(sequence, start, end):
    """Retrieve the sequence from the given sequence string, starting from 'start' position (0-based inclusive) to 'end' position (0-based exclusive)."""
    return "".join(sequence[start:end])  # Ensure the result is a string

def pairwise_alignment(seq1, seq2):
    """Perform pairwise alignment between two sequences using global alignment with affine gap penalties."""
    aligner = get_aligner()
    alignments = aligner.align(seq1, seq2)
    alignment = alignments[0]
    return (str(alignment[0]), str(alignment[1]), alignment.score, 0, len(seq1), len(seq2))

def find_pairwise_points(align, pos):
    pos_pairwise = -1  # Default to -1 if not found
    align_seq1, align_seq2 = align[0], align[1]
    align_pos1, align_pos2 = 0, 0

    for i in range(len(align_seq1)):
        if align_seq1[i] != '-':
            align_pos1 += 1
        if align_seq2[i] != '-':
            align_pos2 += 1
        if align_pos1 == pos:
            pos_pairwise = align_pos2
            break

    return pos_pairwise

def get_unit_sequences(seq, pos, idx, copy_num):
    """Get unit sequences for alignment, returns None if invalid."""
    unit_start, unit_end = find_unit_boundaries(pos, idx)
    if unit_start is None or unit_end is None:
        return None

    pair_start, pair_end = find_nth_unit_after(pos, idx, copy_num)
    if pair_start is None or pair_end is None:
        return None

    unit_seq = get_sequence(seq, unit_start, unit_end)
    pair_seq = get_sequence(seq, pair_start, pair_end)

    if len(unit_seq) == 0 or len(pair_seq) == 0:
        return None

    return unit_start, unit_end, pair_start, pair_end, unit_seq, pair_seq

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

def apply_indel_mutations(seq, generation, pos, records, indel_records):
    """Apply INDEL mutations to sequence."""
    count = 0
    target = np.random.poisson(0.5)

    while count < target:
        idx = random.randint(0, len(seq) - 1)
        indel_type = random.choice(["INS", "DEL"])
        copy_num = np.random.poisson(7.6)

        result = get_unit_sequences(seq, pos, idx, copy_num)
        if result is None:
            continue

        unit_start, unit_end, pair_start, pair_end, unit_seq, pair_seq = result

        try:
            align = pairwise_alignment(unit_seq, pair_seq)
        except IndexError:
            continue

        pair_pos = find_pairwise_points(align, idx - unit_start)
        if pair_pos == -1:
            continue

        pair_abs = int(pair_start) + pair_pos

        if indel_type == "INS":
            ins_seq = "".join(seq[idx:pair_abs])
            records.append((generation, indel_type, idx, seq[idx - 1], "".join(seq[idx - 1:pair_abs]), copy_num))
            seq[idx:idx] = list(ins_seq)
        else:  # DEL
            records.append((generation, indel_type, idx, "".join(seq[idx - 1:pair_abs]), "".join(seq[idx - 1]), copy_num))
            del seq[idx:pair_abs]

        indel_records.append((generation, indel_type, idx, pair_abs))
        count += 1

def get_conversion_type(donor, receipt):
    """Determine conversion type based on sequence comparison."""
    if donor == receipt:
        return "Identical"
    elif len(donor) == len(receipt):
        return "SNP"
    else:
        return "INDEL"

def get_conversion_indel_records(generation, start, end, start_unit_start, start_unit_end,
                               end_unit_start, end_unit_end, start_pair_abs, end_pair_abs,
                               start_pair_unit_start, start_pair_unit_end,
                               end_pair_unit_start, end_pair_unit_end, donor, receipt):
    """Calculate INDEL records for conversions."""
    records = []

    if start_unit_start == end_unit_start and start_pair_unit_start == end_pair_unit_start:
        if len(donor) > len(receipt):
            records.append((generation, "INS", start_pair_abs, start_pair_abs + len(donor) - len(receipt)))
        else:
            records.append((generation, "DEL", end_pair_abs + len(receipt) - len(donor), end_pair_abs))
    else:
        start_donor_len = start_unit_end - start
        start_receipt_len = start_pair_unit_end - start_pair_abs
        if start_donor_len > start_receipt_len:
            records.append((generation, "INS", start_pair_abs, start_pair_abs + start_donor_len - start_receipt_len))
        elif start_donor_len < start_receipt_len:
            records.append((generation, "DEL", start_pair_abs + start_donor_len - start_receipt_len, start_pair_abs))

        end_donor_len = end - end_unit_start
        end_receipt_len = end_pair_abs - end_pair_unit_start
        if end_donor_len > end_receipt_len:
            records.append((generation, "INS", end_pair_abs, end_pair_abs + end_donor_len - end_receipt_len))
        elif end_donor_len < end_receipt_len:
            records.append((generation, "DEL", end_pair_abs + end_donor_len - end_receipt_len, end_pair_abs))

    return records

def apply_conversion_mutations(seq, generation, pos, records, indel_records):
    """Apply conversion mutations to sequence."""
    count = 0
    target = np.random.poisson(1)

    while count < target:
        start = random.randint(0, len(seq) - 1)
        size = np.random.poisson(20)
        end = start + size

        start_unit_start, start_unit_end = find_unit_boundaries(pos, start)
        end_unit_start, end_unit_end = find_unit_boundaries(pos, end)

        if any(x is None for x in [start_unit_start, start_unit_end, end_unit_start, end_unit_end]):
            continue

        start_pair_unit_start, start_pair_unit_end = find_nth_unit_after(pos, start, 1)
        end_pair_unit_start, end_pair_unit_end = find_nth_unit_after(pos, end, 1)

        if any(x is None for x in [start_pair_unit_start, start_pair_unit_end, end_pair_unit_start, end_pair_unit_end]):
            continue

        start_unit_seq = get_sequence(seq, start_unit_start, start_unit_end)
        end_unit_seq = get_sequence(seq, end_unit_start, end_unit_end)
        start_pair_seq = get_sequence(seq, start_pair_unit_start, start_pair_unit_end)
        end_pair_seq = get_sequence(seq, end_pair_unit_start, end_pair_unit_end)

        if any(len(s) == 0 for s in [start_unit_seq, end_unit_seq, start_pair_seq, end_pair_seq]):
            continue

        try:
            align1 = pairwise_alignment(start_unit_seq, start_pair_seq)
            align2 = pairwise_alignment(end_unit_seq, end_pair_seq)
        except IndexError:
            continue

        start_pair = find_pairwise_points(align1, start - start_unit_start)
        end_pair = find_pairwise_points(align2, end - end_unit_start)

        if start_pair == -1 or end_pair == -1:
            continue

        start_pair_abs = int(start_pair_unit_start) + start_pair
        end_pair_abs = int(end_pair_unit_start) + end_pair

        donor = "".join(seq[start:end])
        receipt = "".join(seq[start_pair_abs:end_pair_abs])
        conv_type = get_conversion_type(donor, receipt)

        records.append((generation, "Conversion", start_pair_abs, receipt, donor, conv_type))
        count += 1

        if conv_type == "INDEL":
            conv_indels = get_conversion_indel_records(
                generation, start, end, start_unit_start, start_unit_end,
                end_unit_start, end_unit_end, start_pair_abs, end_pair_abs,
                start_pair_unit_start, start_pair_unit_end,
                end_pair_unit_start, end_pair_unit_end, donor, receipt
            )
            indel_records.extend(conv_indels)

def introduce_mutations(sequence, generation, num_generations, unit_data):
    seq = list(sequence)
    records = []
    pos = np.sort(unit_data["start"].values if isinstance(unit_data, pd.DataFrame) else np.array(unit_data))

    for _ in range(num_generations):
        generation += 1
        indel_records = []

        apply_snp_mutations(seq, generation, records)
        apply_indel_mutations(seq, generation, pos, records, indel_records)
        apply_conversion_mutations(seq, generation, pos, records, indel_records)

        if indel_records:
            pos = np.array(adjust_pos_coordinates(pos.tolist(), indel_records))

    return "".join(seq), records, np.sort(pos).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate mutations in DNA sequences')
    parser.add_argument('input_file', help='Input FASTA file')
    parser.add_argument('num_generations', type=int, help='Number of generations to simulate')
    parser.add_argument('unit_data_file', help='Path to unit data file')
    parser.add_argument('output_file', help='Output FASTA file')
    parser.add_argument('mutation_record', help='Mutation record output file')
    parser.add_argument('adjusted_pos_output', help='Adjusted positions output file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with breakpoint')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    if args.debug == True:
        breakpoint()

    input_file = args.input_file
    match = re.match(r'1\.fasta/(\d+)?generation\.out\.fa', input_file)
    generation = int(match.group(1)) if match else 0
    num_generations = args.num_generations
    unit_data_file = args.unit_data_file
    output_file = args.output_file
    record_output = args.mutation_record
    adjusted_pos_output = args.adjusted_pos_output

    original_sequence = read_sequence(input_file)

    # Read the unit_data file
    unit_data = read_pos_file(unit_data_file)

    mutated_sequence, mutation_records, adjusted_pos = introduce_mutations(original_sequence, generation, num_generations, unit_data)

    with open(output_file, "w") as seq_f:
        seq_f.write(mutated_sequence + '\n')

    with open(record_output, "w") as record_f:
        for record in mutation_records:
            generation, type, idx, ref, mut, copy_num = record
            record_f.write(f"{generation}, {type}, {idx}, {ref}, {mut}, {copy_num}\n")

    with open(adjusted_pos_output, "w") as adjusted_f:
        for pos in sorted(adjusted_pos): # Ensure posiitons are sorted before writing to the file
            adjusted_f.write(f"centro_{generation}gen\t{pos}\n")

    print("Mutated sequence written to", output_file)
    print("Mutation records written to", record_output)
    print("Adjusted positions written to", adjusted_pos_output)

