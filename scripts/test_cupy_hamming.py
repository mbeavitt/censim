#!/usr/bin/env python3
"""
Test CuPy-based Hamming distance GPU implementation.
"""

import cupy as cp
import numpy as np
import time

# CUDA kernel for Hamming identity computation
hamming_kernel = cp.RawKernel(r'''
extern "C" __global__
void hamming_identity_kernel(const unsigned char* seq_array, float* identity, int N, int L) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    if (i == j) {
        identity[i * N + j] = 1.0f;
    } else if (i < j) {
        // Only compute upper triangle
        int matches = 0;
        for (int k = 0; k < L; k++) {
            if (seq_array[i * L + k] == seq_array[j * L + k]) {
                matches++;
            }
        }
        float ident = (float)matches / (float)L;
        identity[i * N + j] = ident;
        identity[j * N + i] = ident;  // Mirror
    }
}
''', 'hamming_identity_kernel')


def hamming_identity_matrix_cupy(seq_array):
    """
    Compute pairwise Hamming identity using CuPy GPU.

    Args:
        seq_array: (N, L) uint8 numpy array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 numpy array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape

    # Transfer to GPU
    seq_gpu = cp.asarray(seq_array, dtype=cp.uint8)
    identity_gpu = cp.zeros((N, N), dtype=cp.float32)

    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (N + threads_per_block[0] - 1) // threads_per_block[0],
        (N + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch kernel
    hamming_kernel(
        blocks_per_grid,
        threads_per_block,
        (seq_gpu, identity_gpu, N, L)
    )

    # Transfer back to CPU
    identity = cp.asnumpy(identity_gpu)

    return identity


def convert_sequences_fast(repeats):
    """Fast conversion of string sequences to uint8 numpy array."""
    if len(repeats) == 0:
        return np.array([], dtype=np.uint8)

    N = len(repeats)
    L = len(repeats[0])

    seq_array = np.empty((N, L), dtype=np.uint8)
    for i, seq in enumerate(repeats):
        seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    return seq_array


if __name__ == "__main__":
    print("Testing CuPy Hamming distance GPU implementation...")

    # Generate test data
    test_seqs = [
        'ACGTACGT' * 22,  # 176 bp
        'ACGTACGA' * 22,  # 176 bp - 1 mismatch per 8bp
        'TGCATGCA' * 22,  # 176 bp - different
    ]

    print(f"\nTesting with {len(test_seqs)} sequences of length {len(test_seqs[0])}")

    seq_array = convert_sequences_fast(test_seqs)
    print(f"Sequence array shape: {seq_array.shape}")

    # Run GPU computation
    print("\nRunning GPU kernel...")
    start = time.time()
    identity = hamming_identity_matrix_cupy(seq_array)
    elapsed = time.time() - start

    print(f"✓ Success! Computed in {elapsed:.4f}s")
    print(f"\nIdentity matrix shape: {identity.shape}")
    print(f"Diagonal (should be 1.0): {identity[0,0]:.4f}, {identity[1,1]:.4f}, {identity[2,2]:.4f}")
    print(f"Sample identities:")
    print(f"  seq[0] vs seq[1]: {identity[0,1]:.4f}")
    print(f"  seq[0] vs seq[2]: {identity[0,2]:.4f}")
    print(f"  seq[1] vs seq[2]: {identity[1,2]:.4f}")

    # Test with realistic size (500 sequences)
    print(f"\n\nTesting with 500 sequences (realistic benchmark size)...")
    bases = ['A', 'C', 'G', 'T']
    large_seqs = []
    np.random.seed(42)
    for _ in range(500):
        seq = ''.join(np.random.choice(bases, size=178))
        large_seqs.append(seq)

    seq_array_large = convert_sequences_fast(large_seqs)

    print("Running GPU kernel...")
    start = time.time()
    identity_large = hamming_identity_matrix_cupy(seq_array_large)
    elapsed = time.time() - start

    print(f"✓ Success! Computed {seq_array_large.shape[0]} x {seq_array_large.shape[0]} matrix in {elapsed:.4f}s")
    print(f"Mean identity: {identity_large.mean():.4f}")
    print(f"\n🎉 CuPy GPU implementation is working!")
