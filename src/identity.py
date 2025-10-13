#!/usr/bin/env python3
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def all_vs_all_identity_naive(repeats):
    """
    Compute all vs all identity matrix using naive nested loops (original method).

    Args:
        repeats: list of sequences (strings)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    # identity cache using hashes
    identity_cache = {}

    # initialize matrix
    mat = np.zeros((n_repeats, n_repeats), dtype=float)

    # Progress reporting
    progress_interval = max(1, n_repeats // 10)

    for i in range(n_repeats):
        if i % progress_interval == 0:
            percent = (i / n_repeats) * 100
            print(f"Progress: {i}/{n_repeats} rows ({percent:.1f}%)")

        for j in range(i, n_repeats):  # upper triangle only
            if i == j:
                identity = 1.0
            else:
                # make a hashable key (order-independent)
                key = tuple(sorted((hash(repeats[i]), hash(repeats[j]))))

                if key not in identity_cache:
                    matches = sum(a == b for a, b in zip(repeats[i], repeats[j]))
                    identity_cache[key] = matches / seq_len

                identity = identity_cache[key]

            mat[i, j] = identity
            mat[j, i] = identity  # symmetry

    print(f"Progress: {n_repeats}/{n_repeats} rows (100.0%)")
    return mat

def all_vs_all_identity(repeats):
    """
    Compute all vs all identity matrix for a list of sequences using vectorized operations.

    Args:
        repeats: list of sequences (strings)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    print(f"Converting sequences to numpy array...")
    # Convert sequences to numpy array for vectorized operations
    # Shape: (n_repeats, seq_len)
    seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)

    print(f"Computing identity matrix...")
    # Use broadcasting to compare all pairs at once
    # seq_array[:, None, :] has shape (n_repeats, 1, seq_len)
    # seq_array[None, :, :] has shape (1, n_repeats, seq_len)
    # Comparison gives shape (n_repeats, n_repeats, seq_len)
    matches = (seq_array[:, None, :] == seq_array[None, :, :])

    # Sum matches along sequence dimension and divide by length
    # Shape: (n_repeats, n_repeats)
    identity_matrix = matches.sum(axis=2) / seq_len

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples to ~max_exact_size sequences and interpolates.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 100)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
        print(f"Computing exact identity ({n_repeats} sequences)...")
    else:
        # Use sqrt-based subsampling for gentler downsampling
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))
        subset_size = n_repeats // subsample_every
        print(f"Computing subsampled identity (every {subsample_every}th sequence, {subset_size}/{n_repeats} total)...")

    if subsample_every == 1:
        # Exact computation
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
    else:
        # Subsampled computation (no interpolation)
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_fast(repeats, sample_positions=None, downsample_factor=1):
    """
    Ultra-fast approximate identity using sampling heuristics.

    Args:
        repeats: list of sequences (strings)
        sample_positions: number of positions to sample (None = use all, faster if < seq_len)
        downsample_factor: only compare every Nth sequence pair (1 = all pairs, 2 = half, etc.)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with approximate pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    # Optionally sample only certain positions in sequences
    if sample_positions is not None and sample_positions < seq_len:
        print(f"Sampling {sample_positions}/{seq_len} positions...")
        indices = np.random.choice(seq_len, size=sample_positions, replace=False)
        sampled_repeats = [''.join(seq[i] for i in indices) for seq in repeats]
        seq_array = np.array([[ord(c) for c in seq] for seq in sampled_repeats], dtype=np.uint8)
    else:
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)

    print(f"Computing approximate identity matrix with scipy.pdist...")
    distances = pdist(seq_array, metric='hamming')
    identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_subsample(repeats, subsample_every=10):
    """
    Compute identity on subsampled sequences, then use for full matrix visualization.

    Args:
        repeats: list of sequences
        subsample_every: only compute identity for every Nth sequence

    Returns:
        Approximate full identity matrix
    """
    n_repeats = len(repeats)

    # Select subset of sequences
    subset_indices = np.arange(0, n_repeats, subsample_every)
    subset_repeats = [repeats[i] for i in subset_indices]

    print(f"Computing identity on {len(subset_repeats)}/{n_repeats} subsampled sequences...")

    # Compute identity on subset
    seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
    distances = pdist(seq_array, metric='hamming')
    subset_matrix = 1 - squareform(distances)

    # Interpolate back to full size using nearest neighbor
    from scipy.ndimage import zoom
    zoom_factor = n_repeats / len(subset_repeats)
    full_matrix = zoom(subset_matrix, zoom_factor, order=0)  # order=0 = nearest neighbor

    return full_matrix.astype(np.float32)

def box_count(matrix, threshold=0.9):
    """
    Calculate fractal dimension using box counting algorithm.

    Args:
        matrix: binary matrix (or continuous matrix to be thresholded)
        threshold: threshold for binarization (default: 0.5)

    Returns:
        fractal_dimension: the estimated fractal dimension
    """
    # Binarize the matrix
    binary_matrix = (matrix > threshold).astype(int)

    # Get matrix dimensions
    n = binary_matrix.shape[0]

    # Box sizes to test (powers of 2 that divide n evenly, plus some others)
    max_box_size = n // 2
    box_sizes = []

    # Generate box sizes: start with powers of 2
    size = 1
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    # Add intermediate sizes for better fitting
    for i in range(len(box_sizes) - 1):
        mid = (box_sizes[i] + box_sizes[i+1]) // 2
        if mid not in box_sizes and mid > box_sizes[i]:
            box_sizes.append(mid)

    box_sizes = sorted(box_sizes)

    print(f"Testing box sizes: {box_sizes}")

    counts = []
    scales = []

    for box_size in box_sizes:
        count = 0
        # Iterate over boxes
        for i in range(0, n, box_size):
            for j in range(0, n, box_size):
                # Extract box
                box = binary_matrix[i:min(i+box_size, n), j:min(j+box_size, n)]
                # Count if box contains any 1s
                if np.any(box):
                    count += 1

        counts.append(count)
        scales.append(1.0 / box_size)
        print(f"Box size {box_size}: {count} boxes")

    # Fit log-log plot
    log_scales = np.log(scales)
    log_counts = np.log(counts)

    # Linear regression
    coeffs = np.polyfit(log_scales, log_counts, 1)
    fractal_dimension = coeffs[0]

    # Calculate R²
    fitted_log_counts = coeffs[0] * log_scales + coeffs[1]
    ss_res = np.sum((log_counts - fitted_log_counts) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nFractal dimension: {fractal_dimension:.4f}")
    print(f"R²: {r_squared:.4f}")

    # Plot the box counting result
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(scales, counts, 'bo-', label='Data')
    fit_counts = np.exp(coeffs[1]) * np.array(scales) ** coeffs[0]
    plt.loglog(scales, fit_counts, 'r--', label=f'Fit: D={fractal_dimension:.4f}, R²={r_squared:.4f}')
    plt.xlabel('Scale (1/box size)')
    plt.ylabel('Number of boxes')
    plt.title('Box Counting Method')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.imshow(binary_matrix, cmap='binary', interpolation='nearest')
    plt.title(f'Binary Matrix (threshold ≥ {threshold})')

    plt.tight_layout()
    plt.savefig('./output/fractal_dimension.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fractal dimension plot saved as ./output/fractal_dimension.png")

    return fractal_dimension

def sliding_window_fractal_dimension(matrix, window_size=100, threshold=0.9):
    """
    Calculate fractal dimension using a sliding window across the matrix.

    Args:
        matrix: binary identity matrix
        window_size: size of the sliding window (default: 100)
        threshold: threshold value for plot labeling

    Returns:
        tuple of (positions, fractal_dimensions) for each window position
    """
    n = matrix.shape[0]

    print(f"\nSliding window analysis:")
    print(f"Matrix size: {n}")
    print(f"Window size: {window_size}")

    if window_size >= n:
        print("Window size >= matrix size, skipping sliding window analysis")
        return [], []

    # Pad the matrix with zeros to allow windows at the edges
    pad_size = window_size // 2
    padded_matrix = np.pad(matrix, pad_size, mode='constant', constant_values=0)
    print(f"Padded matrix size: {padded_matrix.shape[0]} (added {pad_size} on each side)")

    fractal_dimensions = []
    r_squared_values = []
    positions = []

    # Slide the window one repeat at a time
    step = 1
    # Now we can slide from 0 to n (inclusive), using the padded matrix
    for start in range(0, n, step):
        end = start + window_size

        # Extract window from padded matrix
        window = padded_matrix[start:end, start:end]

        # Calculate fractal dimension for this window
        # Use simpler box counting without plotting
        binary_matrix = window
        box_sizes = []
        size = 1
        max_box_size = window_size // 2
        while size <= max_box_size:
            box_sizes.append(size)
            size *= 2

        counts = []
        scales = []

        for box_size in box_sizes:
            count = 0
            for i in range(0, window_size, box_size):
                for j in range(0, window_size, box_size):
                    box = binary_matrix[i:min(i+box_size, window_size), j:min(j+box_size, window_size)]
                    if np.any(box):
                        count += 1
            counts.append(count)
            scales.append(1.0 / box_size)

        if len(scales) > 1:
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            coeffs = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = coeffs[0]

            # Calculate R² for this window
            fitted_log_counts = coeffs[0] * log_scales + coeffs[1]
            ss_res = np.sum((log_counts - fitted_log_counts) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
        else:
            fractal_dim = np.nan
            r_squared = np.nan

        fractal_dimensions.append(fractal_dim)
        r_squared_values.append(r_squared)
        positions.append(start)  # Now we can use the actual position

        if (start // step) % 10 == 0:
            print(f"Progress: window at position {start}/{n}")

    print(f"Computed {len(fractal_dimensions)} windows")

    # Plot results - basic plots
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(positions, fractal_dimensions, 'b-', linewidth=1.5)
    plt.xlabel('Position (repeat index)')
    plt.ylabel('Fractal Dimension')
    plt.title(f'Sliding Window Fractal Dimension\n(window={window_size}, threshold≥{threshold})')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(fractal_dimensions, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Fractal Dimension')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fractal Dimensions')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(positions, r_squared_values, 'r-', linewidth=1.5)
    plt.xlabel('Position (repeat index)')
    plt.ylabel('R²')
    plt.title('Sliding Window R² Values')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(r_squared_values, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('R²')
    plt.ylabel('Frequency')
    plt.title('Distribution of R² Values')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./output/sliding_window_fractal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sliding window plot saved as ./output/sliding_window_fractal.png")

    print(f"\nFractal dimension statistics:")
    print(f"  Mean: {np.mean(fractal_dimensions):.4f}")
    print(f"  Std:  {np.std(fractal_dimensions):.4f}")
    print(f"  Min:  {np.min(fractal_dimensions):.4f}")
    print(f"  Max:  {np.max(fractal_dimensions):.4f}")

    print(f"\nR² statistics:")
    print(f"  Mean: {np.mean(r_squared_values):.4f}")
    print(f"  Std:  {np.std(r_squared_values):.4f}")
    print(f"  Min:  {np.min(r_squared_values):.4f}")
    print(f"  Max:  {np.max(r_squared_values):.4f}")

    # Create combined plot with rotated matrix and sliding window
    from scipy.ndimage import rotate

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(matrix).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 8))

    # Top: rotated matrix with adjusted extent to shift down
    ax1 = plt.subplot(2, 1, 1)
    im = plt.imshow(rotated, cmap='binary', interpolation='nearest', aspect='auto')
    # Shift the image down by adjusting the y limits
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0] * 0.5, ylim[1])
    plt.title(f'Binary Matrix (45° rotation, threshold ≥ {threshold})')
    plt.axis('off')

    # Bottom: sliding window aligned
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(positions, fractal_dimensions, 'b-', linewidth=1.5)
    plt.xlabel('Position (repeat index)')
    plt.ylabel('Fractal Dimension')
    plt.title(f'Sliding Window Fractal Dimension (window={window_size})')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, n)

    plt.tight_layout()
    plt.savefig('./output/combined_matrix_fractal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved as ./output/combined_matrix_fractal.png")

    return positions, fractal_dimensions, r_squared_values

def plot_identity_heatmap(matrix, title="All vs All Sequence Identity", filename="identity_heatmap.png"):
    """
    Create a heatmap visualization of the identity matrix.

    Args:
        matrix: numpy array of pairwise identities
        title: plot title
        filename: output filename
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Identity'},
                rasterized=True)
    plt.title(title)
    plt.xlabel('Repeat Index')
    plt.ylabel('Repeat Index')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Configuration
    identity_threshold = float(sys.argv[1])

    # Load real data
    with open("./data/2191000generation.out.fa", "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)

    # Slice into repeats
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

    import time

    # Compute identity matrix with adaptive subsampling
    print("\n=== Computing identity matrix ===")
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")

    print("\nNumber of repeats:", len(repeats))
    print("Identity Matrix shape:", identity_matrix.shape)
    print(f"Mean identity: {identity_matrix.mean():.3f}")
    matrix_size = identity_matrix.shape[0]
    print(f"Min identity (off-diagonal): {identity_matrix[~np.eye(matrix_size, dtype=bool)].min():.3f}")
    print(f"Max identity (off-diagonal): {identity_matrix[~np.eye(matrix_size, dtype=bool)].max():.3f}")

    # Convert to binary: threshold at identity_threshold
    print(f"\nBinarizing matrix (threshold >= {identity_threshold})...")
    identity_matrix = (identity_matrix >= identity_threshold).astype(float)
    print(f"Binary matrix - fraction of 1s: {identity_matrix.mean():.4f}")

    print("\n=== Computing fractal dimension ===")
    box_count(identity_matrix, threshold=identity_threshold)

    print("\n=== Computing sliding window fractal dimension ===")
    sliding_window_fractal_dimension(identity_matrix, window_size=100, threshold=identity_threshold)

    # Save heatmap
#    print("\nSaving heatmap...")
#    plot_identity_heatmap(identity_matrix,
#                         title=f"All vs All Identity ({n_repeats} repeats, {repeat_len}bp)",
#                         filename="./output/identity_heatmap.png")
#    print("Heatmap saved as ./output/identity_heatmap.png")
