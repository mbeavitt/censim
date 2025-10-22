import random
import numpy as np


class RepeatSequence:
    """Store sequence as a list of repeat units for efficient insertions/deletions.

    Instead of storing 2.67M characters, we store ~15,000 repeat units.
    Insertions/deletions operate on units, which is ~178x faster.
    """
    def __init__(self, sequence, repeat_size=178):
        self.repeat_size = repeat_size
        # Break sequence into repeat units (bytearrays for mutability)
        self.units = []
        for i in range(0, len(sequence), repeat_size):
            self.units.append(bytearray(sequence[i:i+repeat_size], 'ascii'))

    def __len__(self):
        """Return total sequence length in base pairs."""
        return len(self.units) * self.repeat_size

    def num_units(self):
        """Return number of repeat units."""
        return len(self.units)

    def __getitem__(self, idx):
        """Get character at position idx."""
        unit_num = idx // self.repeat_size
        pos_in_unit = idx % self.repeat_size
        return chr(self.units[unit_num][pos_in_unit])

    def __setitem__(self, idx, value):
        """Set character at position idx."""
        unit_num = idx // self.repeat_size
        pos_in_unit = idx % self.repeat_size
        self.units[unit_num][pos_in_unit] = ord(value)

    def insert_units(self, unit_start, unit_end):
        """Insert a copy of units[unit_start:unit_end] at position unit_start.

        This is the key optimization: we copy/insert units, not individual characters.
        """
        units_to_insert = [bytearray(unit) for unit in self.units[unit_start:unit_end]]
        self.units[unit_start:unit_start] = units_to_insert

    def delete_units(self, unit_start, unit_end):
        """Delete units from unit_start to unit_end.

        This is the key optimization: we delete units, not individual characters.
        """
        del self.units[unit_start:unit_end]

    def delete_at_position(self, char_start, char_end):
        """Delete sequence from char_start to char_end at arbitrary positions.

        Example: Delete from A[50] to E[50] across units A,B,C,D,E
        - Takes A[0:50] and E[50:178], merges them into modified A
        - Deletes units B, C, D, E

        Args:
            char_start: Starting character position (inclusive)
            char_end: Ending character position (exclusive)

        Raises:
            ValueError: If deletion size is not a multiple of repeat_size
        """
        deletion_size = char_end - char_start
        if deletion_size % self.repeat_size != 0:
            raise ValueError(
                f"Deletion size must be a multiple of {self.repeat_size}bp to maintain frame alignment. "
                f"Got {deletion_size}bp (char_start={char_start}, char_end={char_end})"
            )

        unit_start = char_start // self.repeat_size
        unit_end = (char_end + self.repeat_size - 1) // self.repeat_size

        pos_start_in_first = char_start % self.repeat_size
        pos_end_in_last = char_end % self.repeat_size

        # Special case: deletion aligns perfectly to unit boundaries
        if pos_start_in_first == 0 and pos_end_in_last == 0:
            # Delete complete units from unit_start to unit_end-1
            del self.units[unit_start:unit_end]
            return

        # Handle single unit case
        if unit_start == unit_end - 1:
            # Deletion within one unit - merge before and after
            before = self.units[unit_start][:pos_start_in_first]
            after = self.units[unit_start][pos_end_in_last:]
            merged = before + after
            if len(merged) != self.repeat_size:
                raise RuntimeError(
                    f"Generated partial unit of size {len(merged)} during single-unit deletion. "
                    f"This indicates a bug in the deletion logic."
                )
            self.units[unit_start] = merged
            return

        # Multi-unit deletion: merge first and last units
        # Take beginning of first unit and end of last unit
        before = self.units[unit_start][:pos_start_in_first]

        if pos_end_in_last > 0:
            after = self.units[unit_end - 1][pos_end_in_last:]
        else:
            # If pos_end_in_last is 0, we take nothing from the last unit
            # (delete ended exactly at unit boundary)
            after = bytearray()
            unit_end -= 1  # Adjust to not delete an extra unit

        # Merge and replace first unit
        merged = before + after
        if len(merged) != self.repeat_size:
            raise RuntimeError(
                f"Generated partial unit of size {len(merged)} during multi-unit deletion. "
                f"This indicates a bug in the deletion logic. "
                f"before={len(before)}, after={len(after)}, deletion_size={deletion_size}"
            )
        self.units[unit_start] = merged

        # Delete all intermediate units (including the last partial unit)
        if unit_end > unit_start + 1:
            del self.units[unit_start + 1:unit_end]

    def duplicate_at_position(self, char_start, char_end):
        """Duplicate sequence from char_start to char_end as a tandem duplication.

        Example: Duplicate A[50] to B[50] across units A, B, C
        - A[50] onwards is A2, A[0:50] is A1
        - B[50] onwards is B2, B[0:50] is B1
        - Result: A1 A2 B1 [A2 B1] B2 C (bracketed = duplicated segment)
        - Maintains 178bp frame alignment

        Args:
            char_start: Starting character position (inclusive)
            char_end: Ending character position (exclusive)

        Raises:
            ValueError: If duplication size is not a multiple of repeat_size
        """
        duplication_size = char_end - char_start
        if duplication_size % self.repeat_size != 0:
            raise ValueError(
                f"Duplication size must be a multiple of {self.repeat_size}bp to maintain frame alignment. "
                f"Got {duplication_size}bp (char_start={char_start}, char_end={char_end})"
            )

        unit_start = char_start // self.repeat_size
        unit_end = (char_end + self.repeat_size - 1) // self.repeat_size

        pos_start_in_first = char_start % self.repeat_size
        pos_end_in_last = char_end % self.repeat_size

        # Build the duplicated segment
        duplicated_units = []

        # Handle single unit case
        if unit_start == unit_end - 1:
            # Duplication within one unit
            # Split into: before | dup_segment | dup_segment | after
            before = self.units[unit_start][:pos_start_in_first]
            dup_segment = self.units[unit_start][pos_start_in_first:pos_end_in_last]
            after = self.units[unit_start][pos_end_in_last:]

            # Create new unit(s) maintaining frame
            # Original: before + dup_segment + after
            # Result: before + dup_segment + dup_segment + after
            self.units[unit_start] = before + dup_segment + dup_segment + after
            return

        # Multi-unit duplication
        # First unit: A1 + A2 (split at char_start)
        first_before = self.units[unit_start][:pos_start_in_first]  # A1
        first_after = self.units[unit_start][pos_start_in_first:]    # A2

        # Last unit: B1 + B2 (split at char_end)
        if pos_end_in_last > 0:
            last_before = self.units[unit_end - 1][:pos_end_in_last]    # B1
            last_after = self.units[unit_end - 1][pos_end_in_last:]     # B2
        else:
            # char_end is at exact unit boundary
            last_before = self.units[unit_end - 1][:]
            last_after = bytearray()
            # Don't need to adjust unit_end since we're including the full last unit

        # Build duplicated segment: A2 + middle_units + B1
        dup_segment = bytearray(first_after)  # A2

        # Add all complete middle units
        for i in range(unit_start + 1, unit_end - 1):
            dup_segment.extend(self.units[i])

        dup_segment.extend(last_before)  # B1

        # Now reconstruct the sequence
        # Original structure: A1|A2 [middle units] B1|B2 [remaining]
        # New structure: A1 A2 B1 [A2 middle B1] B2 [remaining]

        # Build new units maintaining 178bp frame
        new_sequence = first_before + dup_segment + dup_segment + last_after

        # Split new_sequence into 178bp units
        new_units = []
        for i in range(0, len(new_sequence), self.repeat_size):
            unit = bytearray(new_sequence[i:i+self.repeat_size])
            # Ensure all units are exactly repeat_size (pad if last unit is short)
            if len(unit) < self.repeat_size:
                # This should never happen if duplication_size is a multiple of repeat_size
                raise RuntimeError(
                    f"Generated partial unit of size {len(unit)} during duplication. "
                    f"This indicates a bug in the duplication logic."
                )
            new_units.append(unit)

        # Replace affected units
        # Delete old units and insert new ones
        num_old_units = unit_end - unit_start
        self.units[unit_start:unit_end] = new_units

    def get_unit_slice_str(self, unit_start, unit_end):
        """Get string representation of units from unit_start to unit_end."""
        return ''.join(unit.decode('ascii') for unit in self.units[unit_start:unit_end])

    def get_char_slice_str(self, char_start, char_end):
        """Get string representation of characters from char_start to char_end.

        Args:
            char_start: Starting character position (inclusive)
            char_end: Ending character position (exclusive)

        Returns:
            String representation of the character slice
        """
        unit_start = char_start // self.repeat_size
        unit_end = (char_end + self.repeat_size - 1) // self.repeat_size

        pos_start_in_first = char_start % self.repeat_size
        pos_end_in_last = char_end % self.repeat_size

        if unit_start == unit_end - 1:
            # All within one unit
            return self.units[unit_start][pos_start_in_first:pos_end_in_last].decode('ascii')

        # Build string from multiple units
        result = []
        result.append(self.units[unit_start][pos_start_in_first:].decode('ascii'))

        for i in range(unit_start + 1, unit_end - 1):
            result.append(self.units[i].decode('ascii'))

        if pos_end_in_last > 0:
            result.append(self.units[unit_end - 1][:pos_end_in_last].decode('ascii'))
        else:
            result.append(self.units[unit_end - 1].decode('ascii'))

        return ''.join(result)

    def to_string(self):
        """Convert entire sequence to string."""
        return ''.join(unit.decode('ascii') for unit in self.units)


def read_sequence(file_name):
    with open(file_name, "r") as file:
        sequence = file.read().strip()
    return sequence

def apply_snp_mutations(seq, generation, records):
    """Apply SNP mutations to sequence.

    Args:
        seq: RepeatSequence object
        generation: Current generation number
        records: Mutation records list
    """
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
    """Apply INDEL mutations (tandem duplications/deletions) at arbitrary character positions.

    Args:
        seq: RepeatSequence object
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
        seq_length = len(seq)

        # Pick a random character position across the full sequence length
        char_start = random.randint(0, seq_length - 1)

        indel_type = random.choice(["DUP", "DEL"])

        # Sample indel size in base pairs from Poisson distribution
        # Size is always in multiples of repeat_size to maintain frame
        num_repeats = max(1, int(np.random.poisson(7.6)))
        indel_size_bp = num_repeats * repeat_size
        char_end = char_start + indel_size_bp

        # Bounds check: ensure end position is within sequence
        if char_end >= seq_length:
            consecutive_failures += 1
            if consecutive_failures >= max_retries:
                return True  # Signal array collapse
            continue

        # Reset failure counter on success
        consecutive_failures = 0

        # Get string representation for records (before modification)
        indel_seq_str = seq.get_char_slice_str(char_start, char_end)

        if indel_type == "DUP":
            # Tandem duplication at arbitrary position
            prev_base = seq[char_start - 1] if char_start > 0 else ''
            records.append((generation, indel_type, char_start + 1, prev_base,
                          prev_base + indel_seq_str, char_end - char_start))
            seq.duplicate_at_position(char_start, char_end)
        else:  # DEL
            # Deletion at arbitrary position
            prev_base = seq[char_start - 1] if char_start > 0 else ''
            records.append((generation, indel_type, char_start + 1, indel_seq_str,
                          prev_base, char_end - char_start))
            seq.delete_at_position(char_start, char_end)

        indel_records.append((generation, indel_type, char_start, char_end))
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
    # Use RepeatSequence for ~178x faster insertions/deletions
    seq = RepeatSequence(sequence, repeat_size)
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

    return seq.to_string(), records, cenh3_occupancy, collapsed

