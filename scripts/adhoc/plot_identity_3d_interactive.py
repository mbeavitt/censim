#!/usr/bin/env python3
"""
Interactive 3D visualization of sequence identity matrix using Plotly.

Creates two interactive 3D surface plots:
1. Similarity values (identity scores)
2. Distance values (1 - similarity)
"""
import numpy as np
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def matrix_to_3d_points(matrix):
    """
    Convert identity matrix to 3D point coordinates.

    Args:
        matrix: 2D identity matrix (n x n)

    Returns:
        3D array of shape (n*n, 3) with columns [i, j, similarity]
    """
    n = matrix.shape[0]

    # Create coordinate arrays
    i_coords, j_coords = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

    # Flatten and combine
    points_3d = np.column_stack([
        i_coords.ravel(),
        j_coords.ravel(),
        matrix.ravel()
    ])

    return points_3d


def plot_identity_3d_interactive(similarity_matrix, n_repeats, repeat_len, max_points=50000):
    """
    Create interactive 3D point cloud plot using Plotly.

    Args:
        similarity_matrix: numpy array of pairwise identities
        n_repeats: number of repeats
        repeat_len: length of each repeat
        max_points: maximum number of points to plot (default: 50000)
    """
    # Convert to 3D point cloud
    points_3d = matrix_to_3d_points(similarity_matrix)

    print(f"\nPlotly point cloud info:")
    print(f"  Input matrix shape: {similarity_matrix.shape}")
    print(f"  Number of points: {len(points_3d):,}")
    print(f"  Points shape: {points_3d.shape}")
    print(f"  X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"  Z range: [{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}]")
    print(f"  First 5 points:")
    for i in range(min(5, len(points_3d))):
        print(f"    [{points_3d[i, 0]:.0f}, {points_3d[i, 1]:.0f}, {points_3d[i, 2]:.4f}]")
    print(f"  Last 5 points:")
    for i in range(max(0, len(points_3d)-5), len(points_3d)):
        print(f"    [{points_3d[i, 0]:.0f}, {points_3d[i, 1]:.0f}, {points_3d[i, 2]:.4f}]")

    # Create figure
    fig = go.Figure()

    print(f"\nCreating Scatter3d trace...")
    print(f"  X array shape: {points_3d[:, 0].shape}")
    print(f"  Y array shape: {points_3d[:, 1].shape}")
    print(f"  Z array shape: {points_3d[:, 2].shape}")
    print(f"  Any NaN in X? {np.any(np.isnan(points_3d[:, 0]))}")
    print(f"  Any NaN in Y? {np.any(np.isnan(points_3d[:, 1]))}")
    print(f"  Any NaN in Z? {np.any(np.isnan(points_3d[:, 2]))}")

    # Subsample for performance if needed
    if len(points_3d) > max_points:
        stride = len(points_3d) // max_points
        sampled_points = points_3d[::stride]
        print(f"\nSubsampling for browser performance:")
        print(f"  Max points: {max_points:,}")
        print(f"  Original: {len(points_3d):,} points")
        print(f"  Subsampled: {len(sampled_points):,} points (every {stride} points)")
    else:
        sampled_points = points_3d
        print(f"\nUsing all {len(points_3d):,} points (under threshold of {max_points:,})")

    # Add similarity point cloud
    trace = go.Scatter3d(
        x=sampled_points[:, 0],
        y=sampled_points[:, 1],
        z=sampled_points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=sampled_points[:, 2],
            colorscale='Viridis',
            colorbar=dict(
                title='Similarity',
                len=0.75
            ),
            showscale=True,
            opacity=0.6
        ),
        name='Similarity',
        hovertemplate='i: %{x}<br>j: %{y}<br>Similarity: %{z:.4f}<extra></extra>'
    )

    print(f"Adding trace to figure...")
    fig.add_trace(trace)
    print(f"Number of traces in figure: {len(fig.data)}")

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Sequence Similarity (Identity)<br>{n_repeats} repeats, {repeat_len}bp',
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
    """Main function to load data and create interactive 3D visualizations."""

    # Parse command line arguments
    max_points = 50000  # default
    if len(sys.argv) > 1:
        max_points = int(sys.argv[1])
        print(f"Using max_points={max_points:,} from command line")

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
    print("\n=== Creating interactive 3D visualization ===")
    fig = plot_identity_3d_interactive(identity_matrix, n_repeats, repeat_len, max_points=max_points)

    # Save as HTML
    output_file = "./output/identity_3d_interactive.html"
    print(f"\nSaving to {output_file}...")
    fig.write_html(output_file)
    print(f"Interactive plot saved as {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

    print("\n=== Visualization complete ===")
    print("Open the HTML file in your web browser to interact with the 3D plot.")
    print("You can:")
    print("  - Rotate: Click and drag")
    print("  - Zoom: Scroll or pinch")
    print("  - Pan: Right-click and drag")
    print("  - Hover: See exact values at each point")
    print(f"\nUsage: python {sys.argv[0]} [max_points]")
    print(f"  max_points: Maximum number of points to plot (default: 50000)")
    print(f"  Example: python {sys.argv[0]} 100000")


if __name__ == "__main__":
    main()
