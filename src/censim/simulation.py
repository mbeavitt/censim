import random
import numpy as np
from .kmer_wrapper import analyze_repeat_sequence

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

        Args:
            char_start: Starting character position (inclusive)
            char_end: Ending character position (exclusive)

        Raises:
            ValueError: If deletion size is not a multiple of repeat_size
        """
        if (char_end - char_start) % self.repeat_size != 0:
            raise ValueError(
                f"Deletion size must be a multiple of {self.repeat_size}bp to maintain frame alignment. "
                f"Got {char_end - char_start}bp (char_start={char_start}, char_end={char_end})"
            )

        unit_start = char_start // self.repeat_size
        unit_end = char_end // self.repeat_size
        pos_start = char_start % self.repeat_size
        pos_end = char_end % self.repeat_size

        # If deletion aligns to unit boundaries, just delete the units
        if pos_start == 0 and pos_end == 0:
            del self.units[unit_start:unit_end]
            return

        # Merge partial units: keep start of first unit + end of last unit
        before = self.units[unit_start][:pos_start]
        after = self.units[unit_end][pos_end:] if pos_end > 0 else bytearray()
        self.units[unit_start] = before + after

        # Delete remaining units (everything from unit_start+1 to unit_end inclusive)
        del self.units[unit_start + 1:unit_end + 1]

    def duplicate_at_position(self, char_start, char_end):
        """Duplicate sequence from char_start to char_end as a tandem duplication.

        Args:
            char_start: Starting character position (inclusive)
            char_end: Ending character position (exclusive)

        Raises:
            ValueError: If duplication size is not a multiple of repeat_size
        """
        if (char_end - char_start) % self.repeat_size != 0:
            raise ValueError(
                f"Duplication size must be a multiple of {self.repeat_size}bp to maintain frame alignment. "
                f"Got {char_end - char_start}bp (char_start={char_start}, char_end={char_end})"
            )

        # Extract the segment to duplicate as bytes directly (avoiding string conversion)
        unit_start = char_start // self.repeat_size
        unit_end = (char_end + self.repeat_size - 1) // self.repeat_size
        pos_start = char_start % self.repeat_size
        pos_end = char_end % self.repeat_size

        # Build duplicated segment by extracting bytes directly
        dup_segment = bytearray()
        dup_segment.extend(self.units[unit_start][pos_start:])
        for i in range(unit_start + 1, unit_end - 1):
            dup_segment.extend(self.units[i])
        if pos_end > 0:
            dup_segment.extend(self.units[unit_end - 1][:pos_end])
        else:
            dup_segment.extend(self.units[unit_end - 1])

        # Build new sequence: before + dup_segment + dup_segment + after
        before = self.units[unit_start][:pos_start]
        after = self.units[unit_end - 1][pos_end:] if pos_end > 0 else bytearray()
        new_sequence = before + dup_segment + dup_segment + after

        # Split into repeat_size units and replace
        new_units = [bytearray(new_sequence[i:i+self.repeat_size])
                     for i in range(0, len(new_sequence), self.repeat_size)]
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
        pos_start = char_start % self.repeat_size
        pos_end = char_end % self.repeat_size

        # Single unit case
        if unit_start == unit_end - 1:
            return self.units[unit_start][pos_start:pos_end].decode('ascii')

        # Multi-unit case: extract bytes from affected units
        result = bytearray()
        result.extend(self.units[unit_start][pos_start:])
        for i in range(unit_start + 1, unit_end - 1):
            result.extend(self.units[i])
        if pos_end > 0:
            result.extend(self.units[unit_end - 1][:pos_end])
        else:
            result.extend(self.units[unit_end - 1])

        return result.decode('ascii')

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

def d_values_to_position_weights(d_values, seq_length, repeat_size=178, window_size=100, scale_factor=30):
    """Convert D2 values (per window) to position weights (per base pair).

    Args:
        d_values: Array of D2 values at each window position (computed on subsampled repeats)
        seq_length: Length of sequence in base pairs
        repeat_size: Size of each repeat unit (default: 178)
        window_size: Window size used for D2 calculation (default: 100)
        scale_factor: Subsampling scale factor used in identity matrix computation (default: 30)

    Returns:
        weights: Array of weights for each base position (normalized to sum to 1)
    """
    if d_values is None or len(d_values) == 0:
        # Uniform weights if no D2 values
        return np.ones(seq_length) / seq_length

    num_repeats = seq_length // repeat_size

    # Calculate the subsampling factor
    import math
    subsample_every = max(1, int(math.sqrt(num_repeats / scale_factor)))

    # Number of subsampled repeats
    num_subsampled = (num_repeats + subsample_every - 1) // subsample_every

    # Each d_value corresponds to a window center position in SUBSAMPLED repeat units
    # Window positions are 0, 1, 2, ... len(d_values)-1 in subsampled space
    window_positions_subsampled = np.arange(len(d_values))

    # Map back to original repeat unit indices
    # Each subsampled index corresponds to (index * subsample_every) in original space
    window_positions_repeats = window_positions_subsampled * subsample_every

    # Map to base pair positions
    bp_window_positions = window_positions_repeats * repeat_size

    # Interpolate d_values to all base positions
    bp_positions = np.arange(seq_length)

    # Use linear interpolation, extrapolate at edges
    weights = np.interp(bp_positions, bp_window_positions, d_values)

    # Ensure non-negative and normalize
    weights = np.maximum(weights, 0.01)  # Small minimum to avoid zero probability
    weights = weights / np.sum(weights)

    return weights


def apply_indel_mutations(seq, generation, records, indel_records, repeat_size=178, max_retries=5000, d_values=None, d2_bias_strength=1.0):
    """Apply INDEL mutations (tandem duplications/deletions) at arbitrary character positions.

    Args:
        seq: RepeatSequence object
        generation: Current generation number
        records: Mutation records list
        indel_records: INDEL coordinate adjustment records
        repeat_size: Size of each repeat unit in bp (default: 178)
        max_retries: Maximum number of consecutive failures before signaling collapse (default: 5000)
        d_values: Optional D2 values to bias insertion locations (higher D2 = more likely)
        d2_bias_strength: Strength of D2 bias (0=uniform, 1=linear, >1=stronger bias)

    Returns:
        bool: True if array collapsed, False otherwise
    """
    count = 0
    target = np.random.poisson(0.5)
    consecutive_failures = 0

    # Track initial sequence length for weight recalculation
    initial_seq_length = len(seq)
    weights = None
    cached_weights_length = 0

    while count < target:
        seq_length = len(seq)

        # Recompute weights if sequence length has changed or not yet computed
        if d_values is not None and d2_bias_strength > 0:
            if weights is None or seq_length != cached_weights_length:
                weights = d_values_to_position_weights(d_values, seq_length, repeat_size)
                # Apply bias strength: raise weights to power
                weights = weights ** d2_bias_strength
                weights = weights / np.sum(weights)  # Renormalize
                cached_weights_length = seq_length

        # Pick a character position based on D2-weighted probability
        if weights is not None:
            char_start = np.random.choice(seq_length, p=weights)
        else:
            # Uniform random position (original behavior)
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

def compute_correlation_dimension(seq, repeat_len=178, r_min=0.01, r_max=0.5, n_radii=50, window_size=100):
    """Compute k-mer diversity values from a RepeatSequence object using fast C implementation.

    This function now uses the optimized C implementation for computing sliding window
    diversity based on k-mer presence/absence Hamming distances.

    Args:
        seq: RepeatSequence object or string sequence
        repeat_len: Length of each repeat unit (default: 178)
        r_min: Minimum radius value (unused, kept for API compatibility)
        r_max: Maximum radius value (unused, kept for API compatibility)
        n_radii: Number of radius values (unused, kept for API compatibility)
        window_size: Sliding window size (default: 100)

    Returns:
        np.array: Diversity values at each window position
    """
    # Convert string to RepeatSequence if needed
    if isinstance(seq, str):
        seq = RepeatSequence(seq, repeat_size=repeat_len)

    # Use C implementation for fast k-mer diversity computation
    positions, diversity = analyze_repeat_sequence(
        seq,
        window_size=window_size,
        method="consecutive"
    )

    return diversity


def introduce_mutations(sequence, generation, num_generations, repeat_size=178, compute_correlation_dim=True, use_d2_bias=False, d2_bias_strength=1.0):
    """Introduce mutations into sequence over multiple generations.

    Args:
        sequence: Initial DNA sequence string
        generation: Starting generation number
        num_generations: Number of generations to simulate
        repeat_size: Size of each repeat unit in bp (default: 178)
        compute_correlation_dim: Whether to compute correlation dimension (default: True)
        use_d2_bias: Whether to bias insertion locations by D2 values (default: False)
        d2_bias_strength: Strength of D2 bias (0=uniform, 1=linear, >1=stronger, default: 1.0)

    Returns:
        tuple: (mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history, d_values_latest)
            where collapsed is True if the array collapsed to zero, False otherwise
            d_values_history is a list of d_values arrays for each generation (empty if compute_correlation_dim=False)
            d_values_latest is the most recent d_values array (or None if compute_correlation_dim=False)
    """
    # Use RepeatSequence for ~178x faster insertions/deletions
    seq = RepeatSequence(sequence, repeat_size)
    records = []
    d_values_history = []

    # Initialize CENH3 occupancy based on initial sequence length
    num_units = len(sequence) // repeat_size
    cenh3_occupancy = initialize_cenh3_occupancy(num_units)

    # Track latest d_values for biasing
    d_values_latest = None

    collapsed = False
    for gen in range(num_generations):
        generation += 1

        apply_snp_mutations(seq, generation, records)

        # Determine which d_values to use for biasing (if enabled)
        d_values_for_bias = None
        if use_d2_bias and d_values_latest is not None:
            d_values_for_bias = d_values_latest

        # Check if array collapsed during INDEL mutations
        collapsed = apply_indel_mutations(
            seq, generation, records, [], repeat_size,
            d_values=d_values_for_bias,
            d2_bias_strength=d2_bias_strength
        )

        # Compute correlation dimension if enabled
        if compute_correlation_dim:
            d_values_latest = compute_correlation_dimension(seq, repeat_len=repeat_size)
            d_values_history.append(d_values_latest)

        if collapsed:
            print("Simulation complete: Array collapsed to zero")
            break

    return seq.to_string(), records, cenh3_occupancy, collapsed, d_values_history, d_values_latest

