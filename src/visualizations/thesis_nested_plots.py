"""
Plotting functions for Thesis Nested Grover Analysis
"""
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import ast

# Import existing plotting functions
from plot_bias_sweep import (
    plot_bias_sweep_capweight_highlight,
    _load_and_prepare,
    _ordered_bias_labels,
    _compute_curve_stats,
    _setup_latex_style,
)


def setup_latex_style():
    """Configure matplotlib for thesis-quality LaTeX figures."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.size': 14,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
    })


def parse_inner_iterations(x):
    """Parse optimal_inner_iterations column which may be a list string like '[0]' or '[3]'"""
    if pd.isna(x) or x == '' or x == '[]':
        return None
    try:
        parsed = ast.literal_eval(str(x))
        if isinstance(parsed, list) and len(parsed) > 0:
            return max(parsed)
        return 0
    except:
        return None


def load_and_prepare_capweight_data(csv_path):
    """Load capweight data and compute derived columns for both nested and cut."""
    df = pd.read_csv(csv_path)
    df['inner_iter_max'] = df['optimal_inner_iterations'].apply(parse_inner_iterations)
    df['cut_inner_iter_max'] = df['cut_inner_iterations'].apply(parse_inner_iterations)
    df['depth_fraction'] = df['depth'] / df['knapsack_size']
    df['log_ratio'] = np.log2(df['global_gatecost'] / df['nested_gatecost'])
    df['log_ratio_nested'] = df['log_ratio']
    df['log_ratio_cut'] = np.log2(df['global_gatecost'] / df['cut_gatecost'])
    df['inner_is_zero'] = df['inner_iter_max'] == 0
    # Each method has its own inner-iteration count, so it needs its own flag.
    df['inner_is_zero_nested'] = df['inner_is_zero']
    df['inner_is_zero_cut'] = df['cut_inner_iter_max'] == 0
    df_valid = df[df['inner_iter_max'].notna() & df['log_ratio'].notna() & np.isfinite(df['log_ratio'])].copy()
    return df, df_valid


# Per-method column holding the log2 cost ratio and the r_in* == 0 flag.
_METHOD_COLUMNS = {
    'NESTED': ('log_ratio_nested', 'inner_is_zero_nested'),
    'CUT': ('log_ratio_cut', 'inner_is_zero_cut'),
}


# --- Thesis styling constants (TUM corporate colours) ---
TUM_BLUE = "#0065BD"        # Primary TUM blue
TUM_ORANGE = "#E37222"      # Accent orange (for differentiation)
TUM_DARK_BLUE = "#005293"   # Darker blue variant
TUM_LIGHT_BLUE = "#64A0C8"  # Light blue (for fills/bands)
TUM_LIGHT_ORANGE = "#F0B88A"  # Light orange (for fills/bands)
THESIS_RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "Thesis_Results"

# Method color mapping: use TUM Blue for primary, TUM Orange for secondary
METHOD_COLORS = {
    'nested': TUM_BLUE,
    'cut': TUM_ORANGE,
    'bnb': TUM_DARK_BLUE,
    'global': '#333333',  # dark gray for baseline
    'PBC': TUM_BLUE,
    'PCBC': TUM_ORANGE,
}

# Map internal method cost columns to thesis acronyms.
_METHOD_TAG = {
    'nested': 'PBC',
    'cut': 'PCBC',
}

# Compact figure geometry: two of these fit side by side on an A4 text width.
_HALF_PAGE_FIGSIZE = (4.0, 3.0)

# Shared thesis font sizes for axis labels, ticks and legends.
_THESIS_LABELSIZE = 15
_THESIS_TICKSIZE = 13
_THESIS_LEGENDSIZE = 11


def _thesis_bias_label(label):
    """Format a normalized bias label 'inner=<tok>_outer=<tok>' as b_s / b_{n-s}."""
    if isinstance(label, str) and label.startswith("inner=") and "_outer=" in label:
        inner, outer = label[len("inner="):].split("_outer=", 1)
        return rf'$b_s = {inner},\; b_{{n-s}} = {outer}$'
    return label.replace('_', r'\_')


def _thesis_shades(base_hex, n):
    """Return n distinguishable shades anchored on base_hex.

    The ramp runs from the base colour (darkest point) up to a light tint, so
    every shade is at least as bright as base_hex - no dark/muddy tones are
    produced.
    """
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    light = base + (white - base) * 0.62   # lightest tint, still tinted
    cmap = mcolors.LinearSegmentedColormap.from_list('tum', [base, light])
    if n <= 1:
        return [cmap(0.0)]
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def _darken(base_hex, factor=0.65):
    """Return a darker variant of base_hex, used for histogram edge colours."""
    return tuple(np.array(mcolors.to_rgb(base_hex)) * factor)


def plot_figure1_bias_sweep(bias_sweep_path, top_k=5, save_dir=None):
    """Figure 1: Bias Sweep Advantage — thesis styled.

    Renames Nested->PBC and Cut->PCBC, colours curves in TUM blue shades,
    labels the biases as b_s / b_{n-s}, and saves the figure to Thesis_Results.
    """
    _setup_latex_style()
    df, methods, baseline_col = _load_and_prepare(bias_sweep_path, use_gatecost=True)

    if df.empty or not methods:
        print("No data to plot.")
        return

    # Determine correlation type from instance names (for filename tagging only).
    inst = df['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    base_colour = TUM_BLUE  # Always use TUM blue for bias sweep
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    bias_col = 'bias_label_normalized'
    bias_labels = _ordered_bias_labels(df[bias_col].dropna().unique().tolist())

    for cost_col, _, _, _, _ in methods:
        ratio_col = f'log_ratio_{cost_col}'
        method_key = 'nested' if 'nested' in cost_col else 'cut'
        method_tag = _METHOD_TAG[method_key]

        fig, ax = plt.subplots(figsize=(9, 6))

        curves = _compute_curve_stats(df, bias_col, bias_labels, ratio_col,
                                      best_depth_only=True, balance_bins=False)
        if not curves:
            print("No curves to plot.")
            plt.close(fig)
            continue

        # Rank curves by overall mean ratio (higher = better) and keep top-k.
        curve_scores = {blabel: st['mean'].mean() for blabel, st in curves.items()}
        ranked = sorted(curve_scores.items(), key=lambda x: x[1], reverse=True)
        top_ranked = [lbl for lbl, _ in ranked[:top_k]]
        shades = _thesis_shades(base_colour, len(top_ranked))

        highlighted_values = np.concatenate([
            curves[blabel]['mean'].to_numpy() for blabel in top_ranked
        ])
        value_range = highlighted_values.max() - highlighted_values.min()
        y_margin = max(0.05, 0.1 * value_range)
        ax.set_ylim(
            min(-0.05, highlighted_values.min() - y_margin),
            highlighted_values.max() + y_margin,
        )

        # Retain the remaining configurations as context without giving their
        # extreme values control over the plot's y-scale.
        for blabel in bias_labels:
            if blabel not in curves or blabel in top_ranked:
                continue
            st = curves[blabel]
            ax.plot(st['bin_mid'], st['mean'], '-', color='grey',
                    alpha=0.18, linewidth=1.0, zorder=1)

        # Highlighted top-k curves in TUM colour shades.
        for rank, blabel in enumerate(top_ranked):
            st = curves[blabel]
            ax.plot(st['bin_mid'], st['mean'], 'o-', color=shades[rank],
                    label=_thesis_bias_label(blabel),
                    alpha=1.0, linewidth=2.5, markersize=7, zorder=2)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel(r'\textbf{Capacity to Weight Ratio}', fontsize=_THESIS_LABELSIZE)
        ax.set_ylabel(
            rf'\textbf{{Gatecount Ratio}} {{\boldmath$C_{{\mathrm{{rel}}}}^{{\mathrm{{{method_tag}}}}}$}}',
            fontsize=_THESIS_LABELSIZE,
        )
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.legend(loc='lower left', fontsize=_THESIS_LEGENDSIZE, ncol=1, frameon=True,
                  facecolor='white', framealpha=0.9, edgecolor='0.7')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        out_path = save_dir / f'bias_sweep_{method_tag}_{corr_tag}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


def plot_unbiased_capweight(csv_path, save_dir=None, best_depth_only=True, balance_bins=False):
    """Unbiased Capweight Performance — thesis styled.

    Plots the gatecount ratio vs. capacity-to-weight ratio for PBC and PCBC methods.
    One half-page figure saved to Thesis_Results.

    Args:
        csv_path: Path to CSV with capweight multi-method results (unbiased data).
        save_dir: Directory for output (default: Thesis_Results).
        best_depth_only: If True, use only the best depth per instance per method.
        balance_bins: If True, downsample bins to the size of the smallest bin.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Method definitions: (cost_column, label, color, marker)
    method_defs = [
        ('nested_gatecost', 'PBC', TUM_BLUE, 'o'),
        ('cut_gatecost', 'PCBC', TUM_ORANGE, '^'),
    ]
    baseline_col = 'global_gatecost'

    # Coerce relevant cost columns to numeric
    cols_to_coerce = [baseline_col] + [c for c, *_ in method_defs]
    for c in cols_to_coerce:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filter to methods actually present and with valid data
    methods = []
    for cost_col, label, color, marker in method_defs:
        if cost_col in df.columns and (df[cost_col] > 0).any():
            methods.append((cost_col, label, color, marker))

    if not methods:
        print("No method cost columns found in CSV.")
        return

    if baseline_col not in df.columns or not (df[baseline_col] > 0).any():
        print(f"Baseline column '{baseline_col}' missing or has no positive values.")
        return

    # Drop rows where baseline or any selected method column is non-positive
    mask = df[baseline_col] > 0
    for cost_col, *_ in methods:
        mask &= df[cost_col] > 0
    df = df[mask].copy()

    # Compute log ratios for each method
    for cost_col, _, _, _ in methods:
        df[f'log_ratio_{cost_col}'] = np.log2(df[baseline_col] / df[cost_col])

    # Determine correlation type from instance names
    inst = df['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    bins = np.arange(0.05, 1.0, 0.1)

    fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)

    for cost_col, label, color, marker in methods:
        ratio_col = f'log_ratio_{cost_col}'

        # Pick the best depth per instance for this method
        if best_depth_only:
            idx = df.groupby('instance_name')[ratio_col].idxmax()
            mdf = df.loc[idx].reset_index(drop=True)
        else:
            mdf = df.copy()

        mdf['cw_bin'] = pd.cut(mdf['capweight'], bins=bins)
        mdf = mdf.dropna(subset=['cw_bin'])

        if balance_bins:
            bin_counts = mdf.groupby('cw_bin', observed=True).size()
            min_count = int(bin_counts.min())
            if min_count > 0:
                mdf = mdf.groupby('cw_bin', observed=True).apply(
                    lambda g: g.sample(n=min_count, random_state=42)
                ).reset_index(drop=True)

        if mdf.empty:
            continue

        stats_df = mdf.groupby('cw_bin', observed=True).agg(
            mean=(ratio_col, 'mean'),
            std=(ratio_col, 'std'),
            median=(ratio_col, 'median'),
            count=(ratio_col, 'size')
        ).reset_index()
        stats_df['bin_mid'] = stats_df['cw_bin'].apply(lambda b: b.mid).astype(float)
        stats_df = stats_df.sort_values('bin_mid').reset_index(drop=True)
        stats_df['std'] = stats_df['std'].fillna(0)

        ax.fill_between(stats_df['bin_mid'],
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        alpha=0.18, color=color, linewidth=0)
        ax.plot(stats_df['bin_mid'], stats_df['mean'], f'{marker}-', color=color,
                linewidth=2.0, markersize=5, label=rf'\textbf{{{label}}}')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(r'\textbf{Capacity to Weight Ratio}', fontsize=_THESIS_LABELSIZE)
    ax.set_ylabel(
        '\\textbf{Gatecount Ratio}\n'
        '{\\boldmath$C_{\\mathrm{rel}}$}',
        fontsize=_THESIS_LABELSIZE,
    )
    ax.tick_params(labelsize=_THESIS_TICKSIZE)
    ax.legend(loc='upper left', fontsize=_THESIS_LEGENDSIZE, ncol=1, frameon=True,
              facecolor='white', framealpha=0.9, edgecolor='0.7')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Print summary
    print(f"\n=== Unbiased Capweight Performance ({corr_tag}) ===")
    for cost_col, label, _, _ in methods:
        ratio_col = f'log_ratio_{cost_col}'
        if best_depth_only:
            idx = df.groupby('instance_name')[ratio_col].idxmax()
            mdf = df.loc[idx]
        else:
            mdf = df
        win_rate = 100 * (mdf[ratio_col] > 0).mean()
        mean_ratio = mdf[ratio_col].mean()
        print(f"  {label}: win rate = {win_rate:.1f}%, mean ratio = {mean_ratio:.3f}")

    plt.tight_layout()
    out_path = save_dir / f'unbiased_capweight_performance_{corr_tag}.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


def plot_figure2_best_depth_scatter(df_valid):
    """Figure 2: Best Depth per Instance — Colored by Inner Grover Iterations"""
    best_per_instance = df_valid.loc[df_valid.groupby('instance_name')['log_ratio'].idxmax()].copy()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    inner_zero = best_per_instance[best_per_instance['inner_is_zero']]
    inner_positive = best_per_instance[~best_per_instance['inner_is_zero']]
    
    ax.scatter(inner_zero['depth_fraction'], inner_zero['log_ratio'], 
               c=TUM_BLUE, alpha=0.6, s=50, label=f'inner\\_iter=0 (n={len(inner_zero)})', 
               edgecolors=_darken(TUM_BLUE), linewidths=0.5)
    ax.scatter(inner_positive['depth_fraction'], inner_positive['log_ratio'], 
               c=TUM_ORANGE, alpha=0.6, s=50, label=f'inner\\_iter>0 (n={len(inner_positive)})', 
               edgecolors=_darken(TUM_ORANGE), linewidths=0.5)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel(r'\textbf{Best Depth Fraction ($d/n$)}')
    ax.set_ylabel(r'\textbf{$\log_2(\mathrm{global\_gatecost} / \mathrm{nested\_gatecost})$}')
    ax.set_title(r'\textbf{Figure 2: Best Depth per Instance — Colored by Inner Grover Iterations}')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    pct_zero = 100 * len(inner_zero) / len(best_per_instance)
    ax.text(0.02, 0.98, f'At best depth: {pct_zero:.1f}\\% have inner\\_iter=0', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return best_per_instance


def plot_figure3_inner_iter_comparison(df_valid, best_per_instance):
    """Figure 3: Two complementary views on inner Grover iterations"""
    inner_zero_best = best_per_instance[best_per_instance['inner_is_zero']]['log_ratio']
    inner_positive_best = best_per_instance[~best_per_instance['inner_is_zero']]['log_ratio']
    inner_zero_all = df_valid[df_valid['inner_is_zero']]['log_ratio']
    inner_positive_all = df_valid[~df_valid['inner_is_zero']]['log_ratio']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Histogram
    ax1 = axes[0]
    ax1.hist(inner_zero_best, bins=15, alpha=0.7, color=TUM_BLUE, edgecolor=_darken(TUM_BLUE), 
             label=f'inner\\_iter=0 (n={len(inner_zero_best)})')
    ax1.hist(inner_positive_best, bins=8, alpha=0.7, color=TUM_ORANGE, edgecolor=_darken(TUM_ORANGE), 
             label=f'inner\\_iter>0 (n={len(inner_positive_best)})')
    ax1.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.set_xlabel(r'\textbf{Best-case $\log_2(\mathrm{global} / \mathrm{nested})$}')
    ax1.set_ylabel(r'\textbf{Number of Instances}')
    ax1.set_title(r'\textbf{(a) At Best Depth: Most have inner\_iter=0}')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # RIGHT: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([inner_zero_all, inner_positive_all], 
                    tick_labels=[r'inner\_iter=0', r'inner\_iter$>$0'],
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    bp['boxes'][0].set_facecolor(TUM_LIGHT_BLUE)
    bp['boxes'][0].set_edgecolor(_darken(TUM_BLUE))
    bp['boxes'][1].set_facecolor(TUM_LIGHT_ORANGE)
    bp['boxes'][1].set_edgecolor(_darken(TUM_ORANGE))
    
    ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_ylabel(r'\textbf{$\log_2(\mathrm{global} / \mathrm{nested})$}')
    ax2.set_xlabel(r'\textbf{Inner Grover Iterations}')
    ax2.set_title(r'\textbf{(b) At Any Depth: inner\_iter$>$0 $\Rightarrow$ nested loses}')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    med_zero = inner_zero_all.median()
    med_pos = inner_positive_all.median()
    win_zero = 100*(inner_zero_all > 0).mean()
    win_pos = 100*(inner_positive_all > 0).mean()
    stats_text = f'Median (inner_iter=0): {med_zero:.2f}\nMedian (inner_iter>0): {med_pos:.2f}\nWin rate (inner_iter=0): {win_zero:.0f}%\nWin rate (inner_iter>0): {win_pos:.0f}%'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()


def plot_figure2_both(df_valid):
    """Figure 2: Best Depth per Instance for BOTH nested and cut — Colored by Inner Grover Iterations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    best_per_instance_dict = {}
    
    for col, (curve_name, ratio_col) in enumerate([('NESTED', 'log_ratio_nested'), ('CUT', 'log_ratio_cut')]):
        ax = axes[col]
        flag_col = _METHOD_COLUMNS[curve_name][1]
        
        # Filter valid data for this curve
        df_curve = df_valid[df_valid[ratio_col].notna() & np.isfinite(df_valid[ratio_col])].copy()
        best_per_instance = df_curve.loc[df_curve.groupby('instance_name')[ratio_col].idxmax()].copy()
        best_per_instance_dict[curve_name] = best_per_instance
        
        inner_zero = best_per_instance[best_per_instance[flag_col]]
        inner_positive = best_per_instance[~best_per_instance[flag_col]]
        
        ax.scatter(inner_zero['depth_fraction'], inner_zero[ratio_col], 
                   c=TUM_BLUE, alpha=0.6, s=50, label=f'inner\\_iter=0 (n={len(inner_zero)})', 
                   edgecolors=_darken(TUM_BLUE), linewidths=0.5)
        ax.scatter(inner_positive['depth_fraction'], inner_positive[ratio_col], 
                   c=TUM_ORANGE, alpha=0.6, s=50, label=f'inner\\_iter>0 (n={len(inner_positive)})', 
                   edgecolors=_darken(TUM_ORANGE), linewidths=0.5)
        
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_xlabel(r'\textbf{Best Depth Fraction ($d/n$)}')
        method_label = 'nested' if curve_name == 'NESTED' else 'cut'
        ax.set_ylabel(rf'\textbf{{$\log_2(\mathrm{{global}} / \mathrm{{{method_label}}})$}}')
        ax.set_title(rf'\textbf{{{curve_name}: Best Depth per Instance}}')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        pct_zero = 100 * len(inner_zero) / len(best_per_instance)
        ax.text(0.02, 0.98, f'At best depth: {pct_zero:.1f}\\% have inner\\_iter=0', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return best_per_instance_dict


def plot_figure3_both(df_valid, best_per_instance_dict, save_dir=None):
    """Figure 3: Inner iteration histograms at best depth, one figure per method.

    Only the best-depth histograms are produced (the box plots were dropped).
    PBC (nested) and PCBC (cut) are plotted in separate figures and saved to
    Thesis_Results.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    inst = df_valid['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    for curve_name, ratio_col in [('NESTED', 'log_ratio_nested'), ('CUT', 'log_ratio_cut')]:
        best_per_instance = best_per_instance_dict[curve_name]
        method_tag = _METHOD_TAG['nested' if curve_name == 'NESTED' else 'cut']
        flag_col = _METHOD_COLUMNS[curve_name][1]

        inner_zero_best = best_per_instance[best_per_instance[flag_col]][ratio_col]
        inner_positive_best = best_per_instance[~best_per_instance[flag_col]][ratio_col]

        # Shared bin edges so both groups are directly comparable.
        combined = np.concatenate([inner_zero_best.to_numpy(), inner_positive_best.to_numpy()])
        bin_edges = np.histogram_bin_edges(combined, bins=20)

        fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)
        ax.hist(inner_zero_best, bins=bin_edges, alpha=0.75, color=TUM_BLUE,
                edgecolor=_darken(TUM_BLUE), linewidth=0.8,
                label=rf'$r_{{\mathrm{{in}}}}^{{*}} = 0$ ({len(inner_zero_best)} total)')
        ax.hist(inner_positive_best, bins=bin_edges, alpha=0.75, color=TUM_ORANGE,
                edgecolor=_darken(TUM_ORANGE), linewidth=0.8,
                label=rf'$r_{{\mathrm{{in}}}}^{{*}} > 0$ ({len(inner_positive_best)} total)')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_xlabel(
            rf'\textbf{{Gatecount Ratio}} {{\boldmath$C_{{\mathrm{{rel}}}}^{{\mathrm{{{method_tag}}}}}$}}',
            fontsize=_THESIS_LABELSIZE,
        )
        ax.set_ylabel(r'\textbf{Number of Instances}', fontsize=_THESIS_LABELSIZE)
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.32)  # headroom for the legend
        ax.legend(loc='upper left', fontsize=_THESIS_LEGENDSIZE, ncol=1, frameon=True,
                  facecolor='white', framealpha=0.9, edgecolor='0.7')
        ax.grid(True, linestyle='--', alpha=0.3)

        pct_zero = 100 * len(inner_zero_best) / len(best_per_instance)
        print(f"\n=== {method_tag} Summary ===")
        print(f"  r_in*=0 at best depth: {len(inner_zero_best)} ({pct_zero:.1f}%)")
        print(f"  Mean gatecount ratio (r_in*=0): {inner_zero_best.mean():.3f}")
        print(f"  Mean gatecount ratio (r_in*>0): {inner_positive_best.mean():.3f}")

        plt.tight_layout()
        out_path = save_dir / f'inner_iterations_best_depth_{method_tag}_{corr_tag}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()
        plt.close(fig)


def plot_figure3c_conditional_best_depth(df_valid, save_dir=None, x_min=-10.0):
    """Figure 3c: Best depth chosen separately within each r_in regime.

    For every instance two depths are selected: the best depth among all rows
    with r_in* = 0 and the best depth among all rows with r_in* > 0. Both
    distributions are shown in one histogram per method, so the plot isolates
    the cost of forcing inner amplification instead of merely reporting which
    regime happens to win.

    Values below ``x_min`` are collected in the leftmost bin so that the far
    negative outliers do not stretch the axis while the totals stay intact.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    inst = df_valid['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    for curve_name, ratio_col in [('NESTED', 'log_ratio_nested'), ('CUT', 'log_ratio_cut')]:
        method_tag = _METHOD_TAG['nested' if curve_name == 'NESTED' else 'cut']
        flag_col = _METHOD_COLUMNS[curve_name][1]

        df_curve = df_valid[df_valid[ratio_col].notna() & np.isfinite(df_valid[ratio_col])].copy()

        def _best_within(mask):
            sub = df_curve[mask]
            if sub.empty:
                return sub.set_index('instance_name')[ratio_col]
            best = sub.loc[sub.groupby('instance_name')[ratio_col].idxmax()]
            return best.set_index('instance_name')[ratio_col]

        best_zero = _best_within(df_curve[flag_col])
        best_positive = _best_within(~df_curve[flag_col])

        # Share of instances whose overall best depth (over all depths) has r_in* = 0.
        best_overall = df_curve.loc[df_curve.groupby('instance_name')[ratio_col].idxmax()]
        n_instances = len(best_overall)
        pct_overall_zero = 100 * best_overall[flag_col].mean()

        combined = np.concatenate([best_zero.to_numpy(), best_positive.to_numpy()])
        n_clipped = int((combined < x_min).sum())
        lower = max(x_min, combined.min())
        bin_edges = np.histogram_bin_edges(combined[combined >= lower], bins=20,
                                           range=(lower, combined.max()))
        zero_plot = np.clip(best_zero.to_numpy(), lower, None)
        positive_plot = np.clip(best_positive.to_numpy(), lower, None)

        fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)
        ax.hist(zero_plot, bins=bin_edges, alpha=0.75, color=TUM_BLUE,
                edgecolor=_darken(TUM_BLUE), linewidth=0.8,
                label=rf'best $r_{{\mathrm{{in}}}}^{{*}} = 0$ ({len(best_zero)} total)')
        ax.hist(positive_plot, bins=bin_edges, alpha=0.75, color=TUM_ORANGE,
                edgecolor=_darken(TUM_ORANGE), linewidth=0.8,
                label=rf'best $r_{{\mathrm{{in}}}}^{{*}} > 0$ ({len(best_positive)} total)')
        ax.set_xlim(left=lower)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_xlabel(
            rf'\textbf{{Gatecount Ratio}} {{\boldmath$C_{{\mathrm{{rel}}}}^{{\mathrm{{{method_tag}}}}}$}}',
            fontsize=_THESIS_LABELSIZE,
        )
        ax.set_ylabel(r'\textbf{Number of Instances}', fontsize=_THESIS_LABELSIZE)
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.32)  # headroom for the legend
        ax.legend(loc='upper left', fontsize=_THESIS_LEGENDSIZE, ncol=1, frameon=True,
                  facecolor='white', framealpha=0.9, edgecolor='0.7')
        ax.grid(True, linestyle='--', alpha=0.3)

        print(f"\n=== {method_tag} conditional best depth ===")
        print(f"  Overall best depth has r_in*=0 for {pct_overall_zero:.1f}% of {n_instances} instances")
        print(f"  Values below x_min={x_min}: {n_clipped} (collected in the leftmost bin)")
        print(f"  Instances with a r_in*=0 depth: {len(best_zero)}, with a r_in*>0 depth: {len(best_positive)}")
        print(f"  Mean gatecount ratio (best r_in*=0): {best_zero.mean():.3f}")
        print(f"  Mean gatecount ratio (best r_in*>0): {best_positive.mean():.3f}")
        print(f"  Win rate (best r_in*=0): {100 * (best_zero > 0).mean():.1f}%, "
              f"(best r_in*>0): {100 * (best_positive > 0).mean():.1f}%")

        plt.tight_layout()
        out_path = save_dir / f'inner_iterations_conditional_best_depth_{method_tag}_{corr_tag}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()
        plt.close(fig)


def plot_figure4_depth_fraction(depth_fraction_path, save_dir=None, balance_bins=True):
    """Figure 4: Gatecount ratio versus depth fraction — thesis styled.

    One half-page figure per method (PBC / PCBC) showing the binned mean with a
    +/- 1 std band and the median, so the spread of the achievable advantage
    over the depth fraction s/n is visible. No titles, saved to Thesis_Results
    like the other thesis figures.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(depth_fraction_path)
    df['depth_fraction'] = df['depth'] / df['knapsack_size']

    inst = df['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    bins = np.arange(0.05, 1.05, 0.1)

    for method_key, cost_col in [('nested', 'nested_gatecost'), ('cut', 'cut_gatecost')]:
        method_tag = _METHOD_TAG[method_key]

        sub = df[['depth_fraction', 'global_gatecost', cost_col]].apply(pd.to_numeric, errors='coerce')
        sub = sub[(sub['global_gatecost'] > 0) & (sub[cost_col] > 0)].copy()
        if sub.empty:
            print(f"[skip] {method_tag}: no valid gatecost data in this dataset.")
            continue

        sub['log_ratio'] = np.log2(sub['global_gatecost'] / sub[cost_col])
        sub['df_bin'] = pd.cut(sub['depth_fraction'], bins=bins)
        sub = sub.dropna(subset=['df_bin'])
        if sub.empty:
            print(f"[skip] {method_tag}: no rows inside the depth-fraction bins.")
            continue

        if balance_bins:
            min_count = int(sub.groupby('df_bin', observed=True).size().min())
            if min_count > 0:
                sub = sub.groupby('df_bin', observed=True).sample(
                    n=min_count, random_state=42
                ).reset_index(drop=True)

        stats_df = sub.groupby('df_bin', observed=True).agg(
            mean=('log_ratio', 'mean'),
            std=('log_ratio', 'std'),
            median=('log_ratio', 'median'),
            count=('log_ratio', 'size'),
        ).reset_index()
        stats_df['bin_mid'] = stats_df['df_bin'].apply(lambda b: b.mid).astype(float)
        stats_df = stats_df.sort_values('bin_mid').reset_index(drop=True)
        stats_df['std'] = stats_df['std'].fillna(0)

        fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)
        ax.fill_between(stats_df['bin_mid'],
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        alpha=0.18, color=TUM_BLUE, linewidth=0)
        ax.plot(stats_df['bin_mid'], stats_df['mean'], 'o-', color=TUM_BLUE,
                linewidth=2.0, markersize=5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel(r'\textbf{Depth Fraction} {\boldmath$s/n$}', fontsize=_THESIS_LABELSIZE)
        # Two lines: the single-line variant overflows the half-page figure height.
        ax.set_ylabel(
            '\\textbf{Gatecount Ratio}\n'
            f'{{\\boldmath$C_{{\\mathrm{{rel}}}}^{{\\mathrm{{{method_tag}}}}}$}}',
            fontsize=_THESIS_LABELSIZE,
        )
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.3)

        print(f"\n=== {method_tag} depth-fraction bins ({corr_tag}) ===")
        print(f"  Rows per bin: {stats_df['count'].min()}-{stats_df['count'].max()}")
        best_bin = stats_df.loc[stats_df['mean'].idxmax()]
        print(f"  Best mean bin: s/n = {best_bin['bin_mid']:.2f} at {best_bin['mean']:.3f}")

        plt.tight_layout()
        out_path = save_dir / f'depth_fraction_advantage_{method_tag}_{corr_tag}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()
        plt.close(fig)


def analyze_depth_correlations(depth_fraction_path):
    """Correlation analysis for both nested and cut curves."""
    df_depth = pd.read_csv(depth_fraction_path)
    df_depth['depth_fraction'] = df_depth['depth'] / df_depth['knapsack_size']
    df_depth['log_ratio_nested'] = np.log2(df_depth['global_gatecost'] / df_depth['nested_gatecost'])
    df_depth['log_ratio_cut'] = np.log2(df_depth['global_gatecost'] / df_depth['cut_gatecost'])
    
    for curve_name, ratio_col in [('NESTED', 'log_ratio_nested'), ('CUT', 'log_ratio_cut')]:
        print(f"\n=== {curve_name} Curve: Correlation Analysis ===")
        df_valid = df_depth[df_depth[ratio_col].notna() & np.isfinite(df_depth[ratio_col])]
        if df_valid.empty:
            print("  no valid data for this method - skipped")
            continue
        best_depths = df_valid.loc[df_valid.groupby('instance_name')[ratio_col].idxmax()]
        
        for feat in ['capweight', 'knapsack_size']:
            if feat in best_depths.columns and len(best_depths) >= 2:
                r, p = stats.pearsonr(best_depths['depth_fraction'], best_depths[feat])
                print(f"  {feat}: r = {r:.3f}, |r| = {abs(r):.3f}, p = {p:.2e}")
        
        print(f"  Best depth fraction range: [{best_depths['depth_fraction'].min():.2f}, {best_depths['depth_fraction'].max():.2f}]")
        print(f"  Best depth fraction median: {best_depths['depth_fraction'].median():.2f}")
    
    return df_depth


def plot_figure5_candidate_coverage(df_depth, save_dir=None,
                                    candidates_nested=(0.3, 0.4, 0.2, 0.5, 0.1, 0.6),
                                    candidates_cut=(0.4, 0.5, 0.3, 0.6, 0.2, 0.7)):
    """Figure 5: Candidate depth-fraction coverage — thesis styled.

    Produces four separate half-page figures (one per method and panel):

    * distribution of the best winning depth fraction s/n
    * cumulative share of instances that keep an advantage once the candidate
      depth fractions are swept in the given order

    PBC (nested) and PCBC (cut) are plotted separately, all bars in TUM blue,
    without titles, and saved to Thesis_Results like the other thesis figures.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    inst = df_depth['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    def compute_coverage(fractions, df, ratio_col, tolerance=0.05):
        covered = set()
        for frac in fractions:
            mask = (df['depth_fraction'] >= frac - tolerance) & (df['depth_fraction'] <= frac + tolerance) & (df[ratio_col] > 0)
            covered.update(df[mask]['instance_name'].unique())
        return len(covered)

    def _save(fig, name):
        plt.tight_layout()
        out_path = save_dir / f'{name}_{corr_tag}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()
        plt.close(fig)

    for curve_name, ratio_col, candidates in [
        ('NESTED', 'log_ratio_nested', list(candidates_nested)),
        ('CUT', 'log_ratio_cut', list(candidates_cut)),
    ]:
        method_tag = _METHOD_TAG['nested' if curve_name == 'NESTED' else 'cut']

        df_valid = df_depth[df_depth[ratio_col].notna() & np.isfinite(df_depth[ratio_col])]
        if df_valid.empty:
            print(f"[skip] {method_tag}: no valid gatecost data in this dataset.")
            continue
        df_wins = df_valid[df_valid[ratio_col] > 0].copy()
        best_win_depths = df_wins.loc[df_wins.groupby('instance_name')[ratio_col].idxmax()]
        n_instances = df_valid['instance_name'].nunique()

        # --- Panel 1: distribution of the best winning depth fraction ---
        fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)
        ax.hist(best_win_depths['depth_fraction'], bins=10, color=TUM_BLUE,
                edgecolor=_darken(TUM_BLUE), linewidth=0.8, alpha=0.85)
        ax.set_xlabel(r'\textbf{Depth Fraction} {\boldmath$s/n$}', fontsize=_THESIS_LABELSIZE)
        ax.set_ylabel(r'\textbf{Number of Instances}', fontsize=_THESIS_LABELSIZE)
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.3)
        _save(fig, f'best_depth_fraction_distribution_{method_tag}')

        # --- Panel 2: cumulative coverage as candidate fractions are added ---
        coverages = [
            100 * compute_coverage(candidates[:i], df_valid, ratio_col) / n_instances
            for i in range(1, len(candidates) + 1)
        ]

        fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)
        bars = ax.bar(range(len(candidates)), coverages, color=TUM_BLUE,
                      edgecolor=_darken(TUM_BLUE), linewidth=0.8, alpha=0.85)
        ax.set_xticks(range(len(candidates)))
        ax.set_xticklabels([f'$+{c}$' for c in candidates])
        ax.set_xlabel(r'\textbf{Added Depth Fraction} {\boldmath$s/n$}', fontsize=_THESIS_LABELSIZE)
        # Two lines: a single-line label overflows the half-page figure height.
        ax.set_ylabel('\\textbf{Cumulative Coverage}\n\\textbf{of Instances (\\%)}',
                      fontsize=_THESIS_LABELSIZE)
        ax.tick_params(labelsize=_THESIS_TICKSIZE)
        ax.set_ylim(0, 115)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        for bar, cov in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    rf'${cov:.1f}$', ha='center', va='bottom', fontsize=_THESIS_LEGENDSIZE - 2)

        _save(fig, f'depth_candidate_coverage_{method_tag}')

        print(f"\n=== {method_tag} candidate coverage ({corr_tag}) ===")
        print(f"  Candidate depth fractions: {candidates}")
        print(f"  Instances: {n_instances}, with at least one winning depth: {len(best_win_depths)}")
        print(f"  Final coverage: {coverages[-1]:.1f}%")


def plot_figure6_fixed_cost(fixed_cost_path, save_dir=None, exclude_greedy_optimal=True,
                            number_of_identical_samples=4, show_methods=None,
                            max_capweight=None, filename_suffix=None):
    """Figure 6: Fixed Cost Performance — thesis styled.

    Plots termination cost exponent vs. optimality gap for multiple methods.
    One half-page figure saved to Thesis_Results.

    Args:
        fixed_cost_path: Path to CSV with fixed cost multi-method results.
        save_dir: Directory for output (default: Thesis_Results).
        exclude_greedy_optimal: Whether to exclude instances where greedy is optimal.
        number_of_identical_samples: Number of samples to average per datapoint.
        show_methods: List of methods to show, e.g. ['global', 'nested', 'cut'].
                      If None, all available methods are shown.
        max_capweight: If set, filter to instances with capweight <= this value.
        filename_suffix: Custom suffix for output filename (replaces correlation tag).
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fixed_cost_path)
    for col in ['threshold_cost', 'termination_cost_exponent']:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column.")

    if exclude_greedy_optimal and 'greedy_is_optimal' in df.columns:
        df = df[df['greedy_is_optimal'] == False]

    if max_capweight is not None and 'capweight' in df.columns:
        df = df[df['capweight'] <= max_capweight].copy()
        print(f"Filtered to capweight <= {max_capweight}: {len(df)} rows remaining")

    # Detect which methods are present
    has_global = 'approximation_ratio_global' in df.columns and not (df['approximation_ratio_global'] == -1).all()
    has_nested = 'approximation_ratio_nested' in df.columns and not (df['approximation_ratio_nested'] == -1).all()
    has_bnb = 'approximation_ratio_bnb' in df.columns and not (df['approximation_ratio_bnb'] == -1).all()
    has_cut = 'approximation_ratio_cut' in df.columns and not (df['approximation_ratio_cut'] == -1).all()

    # Compute gap_closed normalization: (ratio - greedy) / (1 - greedy)
    if 'approximation_ratio_greedy' not in df.columns:
        raise ValueError("CSV must contain 'approximation_ratio_greedy' column for greedy normalization.")
    greedy_gap = 1.0 - df['approximation_ratio_greedy']
    if has_global:
        df['approximation_ratio_global'] = (df['approximation_ratio_global'] - df['approximation_ratio_greedy']) / greedy_gap
    if has_nested:
        df['approximation_ratio_nested'] = (df['approximation_ratio_nested'] - df['approximation_ratio_greedy']) / greedy_gap
    if has_bnb:
        df['approximation_ratio_bnb'] = (df['approximation_ratio_bnb'] - df['approximation_ratio_greedy']) / greedy_gap
    if has_cut:
        df['approximation_ratio_cut'] = (df['approximation_ratio_cut'] - df['approximation_ratio_greedy']) / greedy_gap

    # Average over identical samples
    if number_of_identical_samples > 1:
        group_cols = ['instance_name', 'termination_cost_exponent']
        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column for sample averaging.")
        avg_cols = []
        if has_global:
            avg_cols.append('approximation_ratio_global')
        if has_nested:
            avg_cols.append('approximation_ratio_nested')
        if has_bnb:
            avg_cols.append('approximation_ratio_bnb')
        if has_cut:
            avg_cols.append('approximation_ratio_cut')
        groups = df.groupby(group_cols)

        def _avg_group(group):
            row = group.iloc[0].copy()
            subset = group.head(number_of_identical_samples)
            for col in avg_cols:
                row[col] = subset[col].mean()
            return row
        df = groups.apply(_avg_group).reset_index(drop=True)

    # Determine correlation type from instance names
    inst = df['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    # Build method configs: (flag, col_name, marker, color, label)
    # Use TUM corporate colours with thesis naming
    methods = []
    if has_global and (show_methods is None or 'global' in show_methods):
        methods.append((True, 'approximation_ratio_global', 'o', '#333333', 'Baseline'))  # dark gray
    if has_nested and (show_methods is None or 'nested' in show_methods):
        methods.append((True, 'approximation_ratio_nested', 's', TUM_BLUE, 'PBC'))
    if has_bnb and (show_methods is None or 'bnb' in show_methods):
        methods.append((True, 'approximation_ratio_bnb', 'D', TUM_DARK_BLUE, 'BnB'))
    if has_cut and (show_methods is None or 'cut' in show_methods):
        methods.append((True, 'approximation_ratio_cut', '^', TUM_ORANGE, 'PCBC'))

    fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)

    # Compute aggregated statistics
    agg_dict = {}
    for _, col_name, _, _, _ in methods:
        agg_dict[col_name] = ['mean', 'std']
    grouped = df.groupby('termination_cost_exponent').agg(agg_dict).reset_index()
    # Flatten columns
    flat_cols = ['termination_cost_exponent']
    for _, col_name, _, _, _ in methods:
        flat_cols.append(f'{col_name}_mean')
        flat_cols.append(f'{col_name}_std')
    grouped.columns = flat_cols

    for _, col_name, marker, color, label in methods:
        ax.errorbar(
            grouped['termination_cost_exponent'],
            grouped[f'{col_name}_mean'],
            yerr=grouped[f'{col_name}_std'],
            fmt=f'{marker}-',
            color=color,
            alpha=0.9,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            markeredgewidth=1.0,
            markeredgecolor='k',
            markersize=6,
            linewidth=1.5,
            label=rf'\textbf{{{label}}}',
        )

    ax.set_xlabel(r'\textbf{Budget Exponent} {\boldmath$\delta$}', fontsize=_THESIS_LABELSIZE)
    ax.set_ylabel(r'\textbf{Optimality Gap} {\boldmath$\eta$}', fontsize=_THESIS_LABELSIZE)
    ax.tick_params(labelsize=_THESIS_TICKSIZE)
    ax.legend(loc='best', fontsize=_THESIS_LEGENDSIZE, ncol=1, frameon=True,
              facecolor='white', framealpha=0.9, edgecolor='0.7')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Print summary statistics
    counts = df.groupby('termination_cost_exponent').size()
    print("\n=== Fixed Cost Performance ===")
    print(f"Correlation type: {corr_tag}")
    print("Instances per termination_cost_exponent:")
    for exponent, count in counts.items():
        print(f"  Exponent {exponent}: {count} instances")

    plt.tight_layout()
    out_suffix = filename_suffix if filename_suffix is not None else corr_tag
    out_path = save_dir / f'fixed_cost_performance_{out_suffix}.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


def _budget_exponent_crossing(deltas, values, target):
    """First budget exponent at which the gap-closed curve reaches ``target``.

    The curve is monotonised (running maximum) before the crossing is located,
    so sampling noise cannot produce a spuriously early crossing. Linear
    interpolation between the two bracketing grid points gives a sub-grid
    resolution. Returns NaN if the target is never reached.
    """
    y = np.maximum.accumulate(np.asarray(values, dtype=float))
    d = np.asarray(deltas, dtype=float)
    reached = y >= target
    if not reached.any():
        return np.nan
    idx = int(np.argmax(reached))
    if idx == 0:
        return d[0]
    y0, y1 = y[idx - 1], y[idx]
    if y1 <= y0:
        return d[idx]
    return d[idx - 1] + (target - y0) / (y1 - y0) * (d[idx] - d[idx - 1])


def _size_bin(sizes, width=5):
    """Group sizes into contiguous fixed-width intervals, merging the upper tail."""
    values = np.asarray(sizes, dtype=int)
    lower = int(values.min())
    upper = int(values.max())
    final_start = lower + width * max(0, (upper - lower - 1) // width)
    return np.minimum(lower + width * ((values - lower) // width), final_start)


def plot_unbiased_size_scaling(fixed_cost_path, method='nested', save_dir=None,
                               targets='auto', exclude_greedy_optimal=True,
                               max_capweight=None, min_size_count=3,
                               auto_grid=np.arange(0.10, 0.96, 0.05), auto_scope='bin',
                               max_censored_fraction=0.02, size_bin_width=5,
                               filename_suffix=None):
    """Budget-exponent saving over the instance size — thesis styled.

    Quantifies the advantage visible in the fixed-budget figures as a single
    number per instance: the difference in budget exponent that the baseline
    and the method need in order to close a given share ``q`` of the optimality
    gap left open by greedy. Because the budget scales as ``const * n**delta``,
    a saving of ``Delta delta`` means the method reaches the same solution
    quality with a budget smaller by a factor ``n**(Delta delta)``.

    Args:
        fixed_cost_path: Path to CSV with fixed cost multi-method results.
        method: 'nested' (PBC) or 'cut' (PCBC).
        save_dir: Directory for output (default: Thesis_Results).
        targets: Gap-closure levels q, or 'auto' to pick the q that maximises
            the mean saving. Selecting q post hoc inflates the effect - see the
            printed scan for the sensitivity across q.
        exclude_greedy_optimal: Whether to drop instances where greedy is optimal.
        max_capweight: If set, keep only instances with capweight <= this value.
        min_size_count: Bins with fewer instances than this are not plotted.
        auto_grid: Candidate q values scanned when ``targets='auto'``.
        auto_scope: 'bin' picks the best q separately within each size bin,
            'global' picks a single q for the whole curve. Per-bin selection
            takes a maximum over the grid in every bin, which biases each point
            upwards by an amount that grows as the bin gets smaller.
        max_censored_fraction: Candidates losing more than this share of
            instances to censoring are excluded from the automatic selection.
        size_bin_width: Instance sizes are grouped into contiguous intervals of
            this width; the final interval includes the largest observed size.
        filename_suffix: Custom suffix for the output filename.
    """
    _setup_latex_style()
    save_dir = pathlib.Path(save_dir) if save_dir is not None else THESIS_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    method_tag = _METHOD_TAG[method]
    method_col = f'approximation_ratio_{method}'

    df = pd.read_csv(fixed_cost_path)
    if exclude_greedy_optimal and 'greedy_is_optimal' in df.columns:
        df = df[df['greedy_is_optimal'] == False]
    if max_capweight is not None and 'capweight' in df.columns:
        df = df[df['capweight'] <= max_capweight]
    if df.empty:
        print("[skip] no rows left after filtering.")
        return

    # Share of the greedy-to-optimum gap that each method closes.
    greedy_gap = 1.0 - df['approximation_ratio_greedy']
    df = df.assign(
        gap_method=(df[method_col] - df['approximation_ratio_greedy']) / greedy_gap,
        gap_base=(df['approximation_ratio_global'] - df['approximation_ratio_greedy']) / greedy_gap,
    )

    # Average the repeated samples of the same (instance, budget exponent).
    per_point = (df.groupby(['instance_name', 'knapsack_size', 'termination_cost_exponent'])
                   [['gap_method', 'gap_base']].mean()
                   .reset_index()
                   .sort_values('termination_cost_exponent'))

    inst = per_point['instance_name'].astype(str)
    is_uncorrelated = inst.str.contains('uncorrelated').mean() >= 0.5
    corr_tag = 'uncorrelated' if is_uncorrelated else 'weakly_correlated'

    auto_select = isinstance(targets, str) and targets == 'auto'
    candidates = [round(float(q), 3) for q in (auto_grid if auto_select else targets)]

    records = []
    for (name, size), sub in per_point.groupby(['instance_name', 'knapsack_size']):
        deltas = sub['termination_cost_exponent'].to_numpy()
        gap_method, gap_base = sub['gap_method'], sub['gap_base']
        for target in candidates:
            d_method = _budget_exponent_crossing(deltas, gap_method, target)
            d_base = _budget_exponent_crossing(deltas, gap_base, target)
            records.append((name, size, target, d_base - d_method))
    saving = pd.DataFrame(records, columns=['instance_name', 'knapsack_size', 'target', 'delta_saving'])
    saving['size_bin'] = _size_bin(saving['knapsack_size'], size_bin_width)

    n_total = saving.groupby('target').size()
    n_total_size = saving.groupby(['size_bin', 'target']).size()
    saving = saving.dropna(subset=['delta_saving'])

    def _bin_stats(frame):
        out = (frame.groupby('size_bin')['delta_saving']
                    .agg(['mean', 'std', 'size']).reset_index())
        out = out[out['size'] >= min_size_count]
        out['bin_mid'] = out['size_bin'].astype(float) + size_bin_width / 2
        if not out.empty:
            last_start = out.iloc[-1]['size_bin']
            last_end = frame.loc[frame['size_bin'] == last_start, 'knapsack_size'].max() + 1
            out.loc[out.index[-1], 'bin_mid'] = (last_start + last_end) / 2
        out['std'] = out['std'].fillna(0)
        return out.sort_values('bin_mid').reset_index(drop=True)

    print(f"\n=== {method_tag} budget-exponent saving ({corr_tag}) ===")
    curves = []

    if auto_select and auto_scope == 'bin':
        agg = (saving.groupby(['size_bin', 'target'])['delta_saving']
                     .agg(['mean', 'std', 'size']))
        kept = agg['size'] / n_total_size.reindex(agg.index)
        agg = agg[(kept >= 1.0 - max_censored_fraction) & (agg['size'] >= min_size_count)]
        if agg.empty:
            raise ValueError("No (size bin, q) combination survives the censoring limit.")
        best_idx = list(agg['mean'].groupby('size_bin').idxmax().dropna())
        stats_df = agg.loc[best_idx].reset_index()
        stats_df['bin_mid'] = stats_df['size_bin'].astype(float) + size_bin_width / 2
        if not stats_df.empty:
            last_start = stats_df.iloc[-1]['size_bin']
            last_end = saving.loc[saving['size_bin'] == last_start, 'knapsack_size'].max() + 1
            stats_df.loc[stats_df.index[-1], 'bin_mid'] = (last_start + last_end) / 2
        stats_df['std'] = stats_df['std'].fillna(0)
        stats_df = stats_df.sort_values('bin_mid').reset_index(drop=True)

        selected = set(best_idx)
        cur = saving[[(b, t) in selected
                      for b, t in zip(saving['size_bin'], saving['target'])]]
        curves.append((stats_df, cur))

        print("  q chosen per size bin:")
        for _, row in stats_df.iterrows():
            lower = int(row['size_bin'])
            print(f"    n in [{lower}, {lower + size_bin_width}): "
                  f"q = {row['target']:.2f}, "
                  f"mean saving {row['mean']:+.3f} over {int(row['size'])} instances")
    else:
        if auto_select:
            scan = saving.groupby('target')['delta_saving'].mean()
            censored = 1.0 - saving.groupby('target').size() / n_total
            usable = scan[censored[scan.index] <= max_censored_fraction]
            if usable.empty:
                raise ValueError("No candidate q survives the censoring limit.")
            best = float(usable.idxmax())
            print("  scan over q (mean saving):")
            for q in scan.index:
                flag = ' <- selected' if q == best else ''
                print(f"    q={q:.2f}: mean saving {scan[q]:+.3f}, "
                      f"censored {100 * censored[q]:.1f}%{flag}")
            targets = (best,)
        for target in targets:
            cur = saving[saving['target'] == target]
            curves.append((_bin_stats(cur), cur))
            print(f"  q={target:g}: mean saving {cur['delta_saving'].mean():+.3f}, "
                  f"{int(n_total[target] - len(cur))} censored")

    base_colour = METHOD_COLORS[method]
    shades = _thesis_shades(base_colour, len(curves))

    fig, ax = plt.subplots(figsize=_HALF_PAGE_FIGSIZE)

    for rank, (stats_df, cur) in enumerate(curves):
        ax.errorbar(
            stats_df['bin_mid'], stats_df['mean'], yerr=stats_df['std'],
            fmt='o-', color=shades[rank], ecolor=shades[rank],
            elinewidth=1.2, capsize=3, capthick=1.2,
            markeredgewidth=0.8, markeredgecolor='k', markersize=5,
            linewidth=1.8,
        )

        r, p = stats.pearsonr(cur['knapsack_size'], cur['delta_saving'])
        print(f"  Pearson r = {r:.3f} (p = {p:.2e}) over {len(cur)} instances")
        print(f"    mean saving smallest bin: {stats_df['mean'].iloc[0]:+.3f}, "
              f"largest bin: {stats_df['mean'].iloc[-1]:+.3f}")

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
    display_bins = curves[0][0]
    starts = display_bins['size_bin'].tolist()
    ends = starts[1:] + [int(saving['knapsack_size'].max()) + 1]
    for index, (start, end) in enumerate(zip(starts, ends)):
        if index % 2 == 0:
            ax.axvspan(start, end, color='0.5', alpha=0.05, zorder=0)
    ax.set_xticks(display_bins['bin_mid'])
    ax.set_xticklabels([
        rf'$[{int(start)}$--${int(end - 1)}]$'
        for start, end in zip(starts, ends)
    ], fontsize=_THESIS_LEGENDSIZE - 1, rotation=90, ha='center', va='top')
    ax.margins(x=0.06)
    ax.set_xlabel(r'\textbf{Knapsack Size} {\boldmath$n$}',
                  fontsize=_THESIS_LABELSIZE)
    # Two lines: the single-line variant overflows the half-page figure height.
    ax.set_ylabel('\\textbf{Max. Improvement}\n'
                  f'{{\\boldmath$\\Delta\\delta^{{\\mathrm{{{method_tag}}}}}$}}',
                  fontsize=_THESIS_LABELSIZE)
    ax.tick_params(labelsize=_THESIS_TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_suffix = filename_suffix if filename_suffix is not None else f'{method_tag}_{corr_tag}'
    out_path = save_dir / f'unbiased_size_scaling_{out_suffix}.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    _results_dir = pathlib.Path(__file__).resolve().parents[2] / "log" / "results"
    _bias_sweep_dir = _results_dir / "capweight_bias_sweep"
    for _id in ("id4", "id5", "id1", "id3"):
        plot_figure1_bias_sweep(
            str(_bias_sweep_dir / f"bias_sweep_capweight_results_{_id}.csv")
        )

    # Figure 6: Fixed Cost Performance
    _fixed_cost_path = (
        _results_dir
        / "fixed_cost_multi_method_gate_based"
        / "fixed_cost_multi_method_results_id1.csv"
    )
    plot_figure6_fixed_cost(str(_fixed_cost_path))

