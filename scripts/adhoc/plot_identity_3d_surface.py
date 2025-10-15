#!/usr/bin/env python3
"""
Interactive 3D surface visualization of sequence identity matrix using Plotly.

Uses Surface plot instead of point cloud for better performance.
"""
import numpy as np
import sys
import plotly.graph_objects as go
import time


def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples sequences and interpolates.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 1000)
        scale_factor: controls subsampling rate for large arrays (default: 30)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    from scipy.spatial.distance import pdist, squareform

    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
        print(f"Computing exact identity ({n_repeats} sequences)...")
    else:
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


def plot_identity_3d_surface(similarity_matrix, n_repeats, repeat_len):
    """
    Create interactive 3D surface plot using Plotly.

    Args:
        similarity_matrix: numpy array of pairwise identities
        n_repeats: number of repeats
        repeat_len: length of each repeat
    """
    n = similarity_matrix.shape[0]

    print(f"\nCreating 3D surface plot:")
    print(f"  Matrix size: {n} x {n}")
    print(f"  Total elements: {n*n:,}")

    # Create coordinate arrays
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)

    # Create figure
    fig = go.Figure()

    # Add surface plot
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=similarity_matrix,
            colorscale='Viridis',
            name='Similarity',
            colorbar=dict(
                title='Similarity',
                len=0.75
            ),
            hovertemplate='i: %{x}<br>j: %{y}<br>Similarity: %{z:.4f}<extra></extra>',
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project=dict(z=True)
                )
            )
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Sequence Similarity (Identity) - Surface Plot<br>{n_repeats} repeats, {repeat_len}bp',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Repeat Index (i)',
            yaxis_title='Repeat Index (j)',
            zaxis_title='Similarity',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=800,
        width=1000,
        showlegend=False
    )

    return fig


def main():
    """Main function to load data and create interactive 3D surface visualization."""

    # Load data
    print("Loading sequence data...")
    with open("./data/2191000generation.out.fa", "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)

    # Slice into repeats
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

    # Compute identity matrix
    print("\n=== Computing identity matrix ===")
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")

    print("\nIdentity Matrix Statistics:")
    print(f"  Shape: {identity_matrix.shape}")
    print(f"  Mean identity: {identity_matrix.mean():.4f}")

    matrix_size = identity_matrix.shape[0]
    off_diagonal_mask = ~np.eye(matrix_size, dtype=bool)
    print(f"  Min identity (off-diagonal): {identity_matrix[off_diagonal_mask].min():.4f}")
    print(f"  Max identity (off-diagonal): {identity_matrix[off_diagonal_mask].max():.4f}")

    # Create output directory if needed
    import os
    os.makedirs('./output', exist_ok=True)

    # Create interactive plot
    print("\n=== Creating interactive 3D surface visualization ===")
    fig = plot_identity_3d_surface(identity_matrix, n_repeats, repeat_len)

    # Save as HTML
    output_file = "./output/identity_3d_surface.html"
    print(f"\nSaving to {output_file}...")
    fig.write_html(output_file)
    print(f"Interactive plot saved as {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

    print("\n=== Visualization complete ===")
    print("Open the HTML file in your web browser to interact with the 3D surface.")
    print("You can:")
    print("  - Rotate: Click and drag")
    print("  - Zoom: Scroll or pinch")
    print("  - Pan: Right-click and drag")
    print("  - Hover: See exact values at each point")
    print("  - The surface includes contour lines projected on the bottom")


if __name__ == "__main__":
    main()
