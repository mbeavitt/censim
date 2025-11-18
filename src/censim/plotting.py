"""
Simplified plotting functions for centromere simulation visualization.
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from censim.identity import all_vs_all_identity_scipy
from censim.correlation_dimension import hamming_distance_matrix


def plot_similarity_and_kmer(sequence, repeat_len, d_values, generation,
                             output_file=None, scale_factor=30, max_exact_size=1000,
                             image_format='jpeg', jpeg_quality=85):
    """
    Create a plot with rotated distance matrix on top and kmer similarity index below.

    Args:
        sequence: DNA sequence string
        repeat_len: Length of repeat units (typically 178bp)
        d_values: Array of kmer similarity (D2) values at each position
        generation: Generation number to display
        output_file: Output filename (required)
        scale_factor: Subsampling factor for identity matrix (default: 30)
        max_exact_size: Maximum size for exact computation (default: 1000)
        image_format: Output format - 'png', 'jpeg', or 'webp' (default: 'jpeg')
        jpeg_quality: JPEG quality 1-100 (default: 85)
    """
    # Parse sequence into repeats
    n_repeats = len(sequence) // repeat_len
    repeats = [sequence[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    # Compute identity matrix
    identity_matrix, subsampled_repeats = all_vs_all_identity_scipy(
        repeats, scale_factor=scale_factor, max_exact_size=max_exact_size,
        return_subsampled=True
    )

    # Convert to distance
    D = hamming_distance_matrix(identity_matrix)
    n = D.shape[0]  # Number of subsampled repeats

    # Pad d_values to match subsampled repeat count if needed
    # d_values is computed with a sliding window, so it's shorter (n - window_size + 1)
    # Pad edges with edge values for proper alignment
    if len(d_values) < n:
        pad_width = (n - len(d_values)) // 2
        d_values = np.pad(d_values, (pad_width, pad_width), mode='edge')
        # Handle odd padding
        if len(d_values) < n:
            d_values = np.pad(d_values, (0, 1), mode='edge')
        elif len(d_values) > n:
            d_values = d_values[:n]

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(D).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with 2 subplots
    fig, (ax_matrix, ax_d) = plt.subplots(2, 1, figsize=(16, 10),
                                          gridspec_kw={'height_ratios': [2, 0.8]})

    # Top: rotated matrix with viridis
    ax_matrix.imshow(rotated, cmap='viridis', interpolation='nearest', aspect='auto',
                     vmin=0, vmax=0.25)  # Clip at 0.25 for better contrast

    # Shift the image down by adjusting the y limits
    ylim = ax_matrix.get_ylim()
    ax_matrix.set_ylim(ylim[0] * 0.5, ylim[1])

    ax_matrix.set_title('Distance Matrix (45° rotation)\nDark = high identity, Light = low identity',
                       fontsize=12, fontweight='bold')
    ax_matrix.axis('off')

    # Add generation number in top left corner
    ax_matrix.text(0.02, 0.98, f'Generation: {generation}',
                  transform=ax_matrix.transAxes,
                  fontsize=10, fontweight='bold',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom: Kmer similarity index plot
    # d_values now matches n (subsampled repeats) after padding
    positions = np.arange(len(d_values))
    ax_d.plot(positions, d_values, 'b-', linewidth=1.5)
    ax_d.set_xlabel('Position (subsampled repeat index)', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('D₂', fontsize=11, fontweight='bold')
    ax_d.set_title('Kmer Similarity Index', fontsize=12, fontweight='bold')
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xlim(0, n)

    # Auto-adjust ylim to data with 5% padding
    if len(d_values) > 0:
        y_min, y_max = np.min(d_values), np.max(d_values)
        y_range = y_max - y_min
        if y_range > 0:
            ax_d.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.tight_layout(pad=0.5)

    # Save to file
    if output_file:
        save_kwargs = {'dpi': 150, 'bbox_inches': None}

        if image_format.lower() in ['jpg', 'jpeg']:
            save_kwargs['format'] = 'jpeg'
            save_kwargs['pil_kwargs'] = {'quality': jpeg_quality, 'optimize': True}
        elif image_format.lower() == 'webp':
            save_kwargs['format'] = 'webp'
            save_kwargs['pil_kwargs'] = {'quality': jpeg_quality}
        else:  # png
            save_kwargs['format'] = 'png'

        plt.savefig(output_file, **save_kwargs)

    # Close the figure to free memory
    plt.close(fig)
