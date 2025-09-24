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
    adjusted_pos1_temp1 = pos1

    for _, ins_del, p1, p2 in pos2:
        adjusted_pos1_temp2 = []
        for pos in adjusted_pos1_temp1:
            if ins_del == "DEL":
                if pos < p1:
                    adjusted_pos1_temp2.append(pos)  # no change to the positions smaller than p1
                elif p1 <= pos < p2:
                    continue  # delete positions between p1 and p2
                else:  # p2 <= pos
                    adjusted_pos1_temp2.append(pos - (p2 - p1))
            elif ins_del == "INS":
                if pos < p1:
                    adjusted_pos1_temp2.append(pos)  # no change to the positions smaller than p1
                elif p1 <= pos < p2:
                    adjusted_pos1_temp2.append(pos)  # insert/duplicate the positions between p1 and p2
                    adjusted_pos1_temp2.append(pos + (p2 - p1))
                else:  # p2 <= pos
                    adjusted_pos1_temp2.append(pos + (p2 - p1))
        adjusted_pos1_temp1 = adjusted_pos1_temp2
    return adjusted_pos1_temp1

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

def introduce_mutations(sequence, generation, num_generations, unit_data):
    mutated_sequence = list(sequence)
    mutation_records = []
    # Convert to sorted NumPy array for fast binary search
    if isinstance(unit_data, pd.DataFrame):
        adjusted_pos = np.sort(unit_data["start"].values)
    else:
        adjusted_pos = np.sort(np.array(adjusted_pos))

    for _ in range(num_generations):
        generation += 1

        actual_snps = 0
        num_snps = np.random.poisson(0.1)
        while actual_snps < num_snps:
            idx = random.randint(0, len(mutated_sequence) - 1)

            bases = ['A', 'T', 'C', 'G']
            bases.remove(mutated_sequence[idx])
            mutated_base = random.choice(bases)
            actual_snps += 1
            mutation_records.append((generation, "SNP", idx + 1, mutated_sequence[idx], mutated_base, "-"))
            mutated_sequence[idx] = mutated_base

        actual_indels = 0
        num_indels = np.random.poisson(0.5)
        while actual_indels < num_indels:
            idx = random.randint(0, len(mutated_sequence) - 1)
            indel_type = random.choice(["INS", "DEL"])

            copy_num = np.random.poisson(7.6)
            idx_unit_start, idx_unit_end = find_unit_boundaries(adjusted_pos, idx)

            if idx_unit_start is None or idx_unit_end is None:
                continue
            idx_unit_seq = get_sequence(mutated_sequence, idx_unit_start, idx_unit_end)

            idx_pairwise_unit_start, idx_pairwise_unit_end = find_nth_unit_after(adjusted_pos, idx, copy_num)

            if idx_pairwise_unit_start is None or idx_pairwise_unit_end is None:
                continue
            idx_pairwise_unit_seq = get_sequence(mutated_sequence, idx_pairwise_unit_start, idx_pairwise_unit_end)

            if len(idx_unit_seq) == 0 or len(idx_pairwise_unit_seq) == 0:
                continue

            try:
                align = pairwise_alignment(idx_unit_seq, idx_pairwise_unit_seq)
            except IndexError:
                continue

            idx_pairwise = find_pairwise_points(align, idx - idx_unit_start)

            if idx_pairwise == -1:  # Skip if pairwise point was not found
                continue

            idx_pairwise_abs = int(idx_pairwise_unit_start) + idx_pairwise

            if indel_type == "INS":
                insertion_sequence = "".join(mutated_sequence[idx:idx_pairwise_abs])
                mutation_records.append((generation, indel_type, idx, mutated_sequence[idx - 1], "".join(mutated_sequence[idx - 1:idx_pairwise_abs]), copy_num))
                mutated_sequence[idx:idx] = list(insertion_sequence)
                actual_indels += 1
            elif indel_type == "DEL":
                mutation_records.append((generation, indel_type, idx, "".join(mutated_sequence[idx - 1:idx_pairwise_abs]), "".join(mutated_sequence[idx - 1]), copy_num))
                del mutated_sequence[idx:idx_pairwise_abs]
                actual_indels += 1

            # Update adjusted_pos
            indel_records = [(generation, indel_type, idx, idx_pairwise_abs)]
            adjusted_pos = np.sort(np.array(adjust_pos_coordinates(adjusted_pos.tolist(), indel_records)))

        actual_conversions = 0
        num_conversions = np.random.poisson(1)
        while actual_conversions < num_conversions:
            start = random.randint(0, len(mutated_sequence) - 1)
            conversion_size = np.random.poisson(20)
            end = start + conversion_size

            start_unit_start, start_unit_end = find_unit_boundaries(adjusted_pos, start)
            end_unit_start, end_unit_end = find_unit_boundaries(adjusted_pos, end)

            if (start_unit_start is None or start_unit_end is None or
                end_unit_start is None or end_unit_end is None):
                continue
            start_unit_seq = get_sequence(mutated_sequence, start_unit_start, start_unit_end)
            end_unit_seq = get_sequence(mutated_sequence, end_unit_start, end_unit_end)

            start_pairwise_unit_start, start_pairwise_unit_end = find_nth_unit_after(adjusted_pos, start, 1)
            end_pairwise_unit_start, end_pairwise_unit_end = find_nth_unit_after(adjusted_pos, end, 1)

            if (start_pairwise_unit_start is None or start_pairwise_unit_end is None or
                end_pairwise_unit_start is None or end_pairwise_unit_end is None):
                continue
            start_pairwise_unit_seq = get_sequence(mutated_sequence, start_pairwise_unit_start, start_pairwise_unit_end)
            end_pairwise_unit_seq = get_sequence(mutated_sequence, end_pairwise_unit_start, end_pairwise_unit_end)

            if len(start_unit_seq) == 0 or len(end_unit_seq) == 0 or len(start_pairwise_unit_seq) == 0 or len(end_pairwise_unit_seq) == 0:
                continue

            try:
                align1 = pairwise_alignment(start_unit_seq, start_pairwise_unit_seq)
                align2 = pairwise_alignment(end_unit_seq, end_pairwise_unit_seq)
            except IndexError:
                continue

            start_pairwise = find_pairwise_points(align1, start - start_unit_start)
            end_pairwise = find_pairwise_points(align2, end - end_unit_start)

            if start_pairwise == -1 or end_pairwise == -1:  # Skip if pairwise point was not found
                continue

            start_pairwise_abs = int(start_pairwise_unit_start) + start_pairwise
            end_pairwise_abs = int(end_pairwise_unit_start) + end_pairwise

            donor_sequence = "".join(mutated_sequence[start:end])
            receipt_sequence = "".join(mutated_sequence[start_pairwise_abs:end_pairwise_abs])
            if donor_sequence == receipt_sequence:
                conversion_out = "Identical"
            elif len(donor_sequence) == len(receipt_sequence):
                conversion_out = "SNP"
            else:
                conversion_out = "INDEL"
            mutation_records.append((generation, "Conversion", start_pairwise_abs, receipt_sequence, donor_sequence, conversion_out))

            actual_conversions += 1

            # Update adjusted_pos
            conversion_indel_records = []
            if conversion_out == "INDEL":
                if start_unit_start == end_unit_start and start_pairwise_unit_start == end_pairwise_unit_start:
                    if len(donor_sequence) > len(receipt_sequence): #INS
                        conversion_indel_records = [(generation, "INS", start_pairwise_abs, start_pairwise_abs + len(donor_sequence) - len(receipt_sequence))]
                    else: #DEL
                        conversion_indel_records = [(generation, "DEL", end_pairwise_abs + len(receipt_sequence) - len(donor_sequence), end_pairwise_abs)]
                else:
                    if (start_unit_end - start) > (start_pairwise_unit_end - start_pairwise_abs): #INS
                        conversion_indel_records = [(generation, "INS", start_pairwise_abs, start_pairwise_abs + (start_unit_end - start) - (start_pairwise_unit_end - start_pairwise_abs))]
                    elif (start_unit_end - start) < (start_pairwise_unit_end - start_pairwise_abs): #DEL
                        conversion_indel_records = [(generation, "DEL", start_pairwise_abs + (start_unit_end - start) - (start_pairwise_unit_end - start_pairwise_abs), start_pairwise_abs)]
                    if (end - end_unit_start) > (end_pairwise_abs - end_pairwise_unit_start): #INS
                        conversion_indel_records = [(generation, "INS", end_pairwise_abs, end_pairwise_abs + (end - end_unit_start) - (end_pairwise_abs - end_pairwise_unit_start))]
                    elif (end - end_unit_start) < (end_pairwise_abs - end_pairwise_unit_start): #DEL
                        conversion_indel_records = [(generation, "DEL", end_pairwise_abs + (end - end_unit_start) - (end_pairwise_abs - end_pairwise_unit_start), end_pairwise_abs)]

                adjusted_pos = np.sort(np.array(adjust_pos_coordinates(adjusted_pos.tolist(), conversion_indel_records)))

    return "".join(mutated_sequence), mutation_records, adjusted_pos.tolist()

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

