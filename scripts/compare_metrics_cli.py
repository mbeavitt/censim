#!/usr/bin/env python3
"""
Compare complexity metrics with CLI support.

This is a template showing how to conditionally compute metrics.
Copy this logic back into compare_fractal_vs_correlation.py
"""

# After loading data and computing identity matrix...

# Initialize storage for computed metrics
computed_metrics = {}

# Conditionally compute FD
if 'fd' in metrics_to_compute:
    print(f"\n3. Computing Fractal Dimension (metric only)...")
    # ... FD computation code ...
    computed_metrics['fd'] = {
        'positions': fd_positions_scaled,
        'values': fractal_dimensions,
        'label': 'Fractal Dimension',
        'color': 'b'
    }

# Conditionally compute CD
if 'cd' in metrics_to_compute:
    print(f"\n4. Computing Correlation Dimension (metric only)...")
    # ... CD computation code ...
    computed_metrics['cd'] = {
        'positions': d2_positions_scaled,
        'values': correlation_dimensions,
        'label': 'Correlation Dimension (D2)',
        'color': 'r',
        'invert': True  # Mark for inversion
    }

# Conditionally compute k-mer
if 'kmer' in metrics_to_compute:
    print(f"\n5. Computing k-mer Entropy...")
    # ... kmer computation code ...
    computed_metrics['kmer'] = {
        'positions': kmer_positions_scaled,
        'values': kmer_values,
        'label': 'k-mer Entropy',
        'color': 'g',
        'invert': True
    }

# Conditionally compute LZMA
if 'lzma' in metrics_to_compute:
    print(f"\n6. Computing LZMA Compressibility...")
    # ... LZMA computation code ...
    computed_metrics['lzma'] = {
        'positions': lzma_positions_scaled,
        'values': lzma_values,
        'label': 'LZMA Compressibility',
        'color': 'm',
        'invert': True
    }

# Normalize and plot
print(f"\nNormalizing {len(computed_metrics)} metrics...")
for key, metric_data in computed_metrics.items():
    values = metric_data['values']
    if metric_data.get('invert', False):
        values = [-v for v in values]
    metric_data['normalized'] = normalize(values)

# Plot
for key, metric_data in computed_metrics.items():
    ax.plot(metric_data['positions'], metric_data['normalized'],
            color=metric_data['color'], linewidth=1.5,
            label=metric_data['label'], alpha=0.8)
