import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# TUM corporate colours
TUM_BLUE = "#0065BD"
TUM_ORANGE = "#E37222"


def _tum_shades(base_hex, n):
    """Return n distinguishable shades anchored on base_hex.

    The ramp runs from the base colour (darkest point) up to a light tint.
    """
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    light = base + (white - base) * 0.62
    cmap = mcolors.LinearSegmentedColormap.from_list('tum', [base, light])
    if n <= 1:
        return [cmap(0.0)]
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


_BIAS_FRACTION_ORDER = ["0", "n/10", "n/7", "n/4", "n/2", "n"]
_TARGET_FRACTIONS = {
    "0": 0.0,
    "n/10": 0.1,
    "n/7": 1 / 7,
    "n/4": 0.25,
    "n/2": 0.5,
    "n": 1.0,
}


def _parse_new_bias_label(label: str):
    """Parse labels of the form 'inner=<token>_outer=<token>' into token pairs."""
    if not isinstance(label, str):
        return None


def _closest_fraction_label(value: float) -> str:
    return min(_TARGET_FRACTIONS.keys(), key=lambda k: abs(value - _TARGET_FRACTIONS[k]))


def _normalize_bias_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Build a size-normalized symbolic bias label for consistent cross-size grouping."""
    df = df.copy()

    if {'nested_bias_inner', 'nested_bias_outer', 'knapsack_size'}.issubset(df.columns):
        nb_in = pd.to_numeric(df['nested_bias_inner'], errors='coerce')
        nb_out = pd.to_numeric(df['nested_bias_outer'], errors='coerce')
        n = pd.to_numeric(df['knapsack_size'], errors='coerce')
        valid = (n > 0) & nb_in.notna() & nb_out.notna()

        symbolic = []
        for idx in df.index:
            if valid.loc[idx]:
                in_frac = float(nb_in.loc[idx] / n.loc[idx])
                out_frac = float(nb_out.loc[idx] / n.loc[idx])
                in_label = _closest_fraction_label(in_frac)
                out_label = _closest_fraction_label(out_frac)
                symbolic.append(f"inner={in_label}_outer={out_label}")
            else:
                symbolic.append(None)
        df['bias_label_normalized'] = symbolic
    else:
        df['bias_label_normalized'] = None

    for idx in df.index:
        if isinstance(df.at[idx, 'bias_label_normalized'], str):
            continue
        raw = df.at[idx, 'bias_label'] if 'bias_label' in df.columns else None
        pair = _parse_new_bias_label(raw)
        if pair is None:
            df.at[idx, 'bias_label_normalized'] = raw
            continue

        inner_token, outer_token = pair
        try:
            in_label = _closest_fraction_label(float(inner_token))
            out_label = _closest_fraction_label(float(outer_token))
            df.at[idx, 'bias_label_normalized'] = f"inner={in_label}_outer={out_label}"
        except ValueError:
            df.at[idx, 'bias_label_normalized'] = raw

    return df
    if not label.startswith("inner=") or "_outer=" not in label:
        return None
    try:
        inner_str, outer_str = label[len("inner="):].split("_outer=", 1)
        return inner_str, outer_str
    except (ValueError, TypeError):
        return None


def _fraction_rank(token: str):
    if token in _BIAS_FRACTION_ORDER:
        return _BIAS_FRACTION_ORDER.index(token)
    return len(_BIAS_FRACTION_ORDER)


def _ordered_bias_labels(labels):
    """Order bias labels by (inner_bias, outer_bias) for stable, readable legends."""
    parsed_symbolic = []
    parsed_numeric = []
    fallback = []
    for label in labels:
        pair = _parse_new_bias_label(label)
        if pair is None:
            fallback.append(label)
        else:
            inner_token, outer_token = pair
            if inner_token in _BIAS_FRACTION_ORDER and outer_token in _BIAS_FRACTION_ORDER:
                parsed_symbolic.append((_fraction_rank(inner_token), _fraction_rank(outer_token), label))
            else:
                try:
                    parsed_numeric.append((float(inner_token), float(outer_token), label))
                except ValueError:
                    fallback.append(label)

    parsed_symbolic.sort(key=lambda x: (x[0], x[1]))
    parsed_numeric.sort(key=lambda x: (x[0], x[1]))
    fallback.sort()
    return [label for _, _, label in parsed_symbolic] + [label for _, _, label in parsed_numeric] + fallback


def _tex_label(label: str) -> str:
    """Format labels for display; supports new 'inner=..._outer=...' format."""
    pair = _parse_new_bias_label(label)
    if pair is not None:
        inner, outer = pair
        return rf'$b_{{\mathrm{{in}}}}={inner},\;b_{{\mathrm{{out}}}}={outer}$'
    return label.replace('_', r'\_')


def _setup_latex_style():
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
        'legend.fontsize': 13,
    })


def _load_and_prepare(csv_path, use_gatecost=False):
    """Load capweight bias-sweep CSV, detect methods, compute log ratios."""
    df = pd.read_csv(csv_path)

    method_defs = [
        ('nested_cost', 'nested_bias_inner', r'\textbf{Nested}', TUM_BLUE, 'o'),
        ('cut_cost', 'cut_bias_inner', r'\textbf{Cut}', TUM_ORANGE, 'D'),
    ]
    baseline_col = 'global_cost'

    if use_gatecost:
        method_defs = [
            (cost_col.replace('_cost', '_gatecost'), bias_col, label, color, marker)
            for cost_col, bias_col, label, color, marker in method_defs
        ]
        baseline_col = 'global_gatecost'

    cols_to_coerce = [baseline_col] + [c for c, *_ in method_defs]
    for c in cols_to_coerce:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    methods = []
    for cost_col, bias_col, label, color, marker in method_defs:
        if cost_col in df.columns and (df[cost_col] > 0).any():
            methods.append((cost_col, bias_col, label, color, marker))

    mask = df[baseline_col] > 0
    for cost_col, *_ in methods:
        mask &= df[cost_col] > 0
    df = df[mask].copy()

    for cost_col, _, _, _, _ in methods:
        df[f'log_ratio_{cost_col}'] = np.log2(df[baseline_col] / df[cost_col])

    df = _normalize_bias_labels(df)

    return df, methods, baseline_col


def _compute_curve_stats(df, bias_col, bias_labels, ratio_col, best_depth_only, balance_bins):
    """Compute per-bias-label curve statistics. Returns dict: label -> stats DataFrame."""
    curves = {}
    for blabel in bias_labels:
        sub = df[df[bias_col] == blabel].copy()

        if best_depth_only:
            ref_idx = sub.groupby('instance_name')[ratio_col].idxmax()
            sub = sub.loc[ref_idx].reset_index(drop=True)

        bins = np.arange(0.05, 1.0, 0.1)
        sub['cw_bin'] = pd.cut(sub['capweight'], bins=bins)
        sub = sub.dropna(subset=['cw_bin'])

        if balance_bins and not sub.empty:
            bc = sub.groupby('cw_bin', observed=True).size()
            mc = int(bc.min())
            if mc > 0:
                sub = sub.groupby('cw_bin', observed=True).apply(
                    lambda g: g.sample(n=mc, random_state=42)
                ).reset_index(drop=True)

        if sub.empty:
            continue

        stats = sub.groupby('cw_bin', observed=True).agg(
            mean=(ratio_col, 'mean'),
        ).reset_index()
        stats['bin_mid'] = stats['cw_bin'].apply(lambda b: b.mid).astype(float)
        stats = stats.sort_values('bin_mid').reset_index(drop=True)
        curves[blabel] = stats
    return curves


def plot_bias_sweep_capweight(csv_path, best_depth_only=True,
                              balance_bins=True, use_gatecost=False, name_of_file=None):
    """Single plot per method: one curve per bias value, x = binned capweight,
    y = mean log2(global/method) cost."""
    _setup_latex_style()
    df, methods, baseline_col = _load_and_prepare(csv_path, use_gatecost)

    if df.empty or not methods:
        print("No data to plot.")
        return

    bias_col = 'bias_label_normalized'
    bias_labels = _ordered_bias_labels(df[bias_col].dropna().unique().tolist())
    cmap = _tum_shades(TUM_BLUE, max(len(bias_labels), 1))

    for cost_col, _, label, _, _ in methods:
        ratio_col = f'log_ratio_{cost_col}'
        fig, ax = plt.subplots(figsize=(8, 5))

        curves = _compute_curve_stats(df, bias_col, bias_labels, ratio_col,
                                      best_depth_only, balance_bins)

        for i, blabel in enumerate(bias_labels):
            if blabel not in curves:
                continue
            stats = curves[blabel]
            ax.plot(stats['bin_mid'], stats['mean'], 'o-', color=cmap[i],
                    label=_tex_label(blabel), alpha=0.85)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel(r'\textbf{Capacity to Weight Ratio}')
        cost_label = 'gatecost' if use_gatecost else 'cost'
        cost_col_tex = cost_col.replace('_', r'\_')
        ax.set_ylabel(rf'{{\boldmath$\log_2(\mathrm{{global\_{cost_label}}}\;/\;\mathrm{{{cost_col_tex}}})$}}')
        ax.set_title(rf'{label} — Bias Sweep (capweight)')
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        if name_of_file:
            plt.savefig(f"{name_of_file}_{cost_col}.pdf", bbox_inches='tight')
        plt.show()


def plot_bias_sweep_capweight_highlight(csv_path, best_depth_only=True,
                                        balance_bins=False, use_gatecost=True,
                                        name_of_file=None, top_k=3):
    """Like plot_bias_sweep_capweight but larger figure, legend outside, and
    the top-k best curves are highlighted while the rest are dimmed."""
    _setup_latex_style()
    df, methods, baseline_col = _load_and_prepare(csv_path, use_gatecost)

    if df.empty or not methods:
        print("No data to plot.")
        return

    bias_col = 'bias_label_normalized'
    bias_labels = _ordered_bias_labels(df[bias_col].dropna().unique().tolist())
    cmap = _tum_shades(TUM_BLUE, max(len(bias_labels), 1))

    for cost_col, _, label, _, _ in methods:
        ratio_col = f'log_ratio_{cost_col}'
        fig, ax = plt.subplots(figsize=(12, 7))

        curves = _compute_curve_stats(df, bias_col, bias_labels, ratio_col,
                                      best_depth_only, balance_bins)

        if not curves:
            print("No curves to plot.")
            plt.close(fig)
            continue

        # Rank curves by overall mean ratio (higher = better)
        curve_scores = {blabel: stats['mean'].mean() for blabel, stats in curves.items()}
        ranked = sorted(curve_scores.items(), key=lambda x: x[1], reverse=True)
        top_labels = {lbl for lbl, _ in ranked[:top_k]}

        print(f"\n{'='*60}")
        print(f"  {label} — Top {top_k} bias configurations (by mean log2 ratio)")
        print(f"{'='*60}")
        for rank, (blabel, score) in enumerate(ranked[:top_k], 1):
            print(f"  #{rank}: {blabel:40s}  mean_log2_ratio = {score:.4f}")
        print(f"{'='*60}\n")

        # Plot dimmed curves first, then highlighted on top
        for i, blabel in enumerate(bias_labels):
            if blabel not in curves or blabel in top_labels:
                continue
            stats = curves[blabel]
            ax.plot(stats['bin_mid'], stats['mean'], '-', color='grey',
                    alpha=0.25, linewidth=1.0)

        for rank, (blabel, score) in enumerate(ranked[:top_k]):
            if blabel not in curves:
                continue
            i = bias_labels.index(blabel)
            stats = curves[blabel]
            ax.plot(stats['bin_mid'], stats['mean'], 'o-', color=cmap[i],
                    label=_tex_label(blabel) + rf' (rank {rank+1})',
                    alpha=1.0, linewidth=2.5, markersize=7)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel(r'\textbf{Capacity to Weight Ratio}')
        cost_label = 'gatecost' if use_gatecost else 'cost'
        cost_col_tex = cost_col.replace('_', r'\_')
        ax.set_ylabel(rf'{{\boldmath$\log_2(\mathrm{{global\_{cost_label}}}\;/\;\mathrm{{{cost_col_tex}}})$}}')
        ax.set_title(rf'{label} — Bias Sweep (capweight) — Top {top_k} highlighted')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=11, borderaxespad=0)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 0.78, 1])
        if name_of_file:
            plt.savefig(f"{name_of_file}_{cost_col}.pdf", bbox_inches='tight')
        plt.show()
