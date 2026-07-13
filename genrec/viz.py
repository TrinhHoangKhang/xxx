import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_codebook_heatmap(counts: np.ndarray, title: str, save_path: str) -> None:
    """
    Plot a codebook utilization heatmap and save it to disk.

    Args:
        counts   : (n_digits, codebook_size) integer array — number of items assigned
                   to each codeword slot per sub-codebook.
        title    : Figure title (model name + conditions + snapshot label).
        save_path: Full path (including filename) where the PNG is written.
    """
    n_digits, codebook_size = counts.shape

    # Normalize each row (codebook) to percentage so all rows are on the same scale.
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
    pct = counts / row_sums * 100.0

    # Also compute per-row entropy as a summary statistic (logged to console).
    uniform_pct = 100.0 / codebook_size
    prob = pct / 100.0 + 1e-12
    entropy_per_codebook = -(prob * np.log2(prob)).sum(axis=1)   # (n_digits,)
    max_entropy = np.log2(codebook_size)
    mean_utilization = entropy_per_codebook.mean() / max_entropy  # 1.0 = perfectly uniform

    fig_w = max(16, codebook_size // 10)
    fig_h = max(6,  n_digits   //  4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pct, aspect='auto', cmap='YlOrRd', vmin=0, vmax=pct.max())

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Usage per codebook (%)', fontsize=10)

    ax.set_xlabel('Codeword Index', fontsize=11)
    ax.set_ylabel('Codebook Index', fontsize=11)
    ax.set_title(
        f'{title}\n(mean codebook utilization: {mean_utilization * 100:.1f}% of max entropy)',
        fontsize=12, fontweight='bold'
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, codebook_size // 8)))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(max(1, n_digits // 8)))

    plt.tight_layout()
    dir_part = os.path.dirname(save_path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIZ] Saved heatmap → {save_path}  (utilization: {mean_utilization * 100:.1f}%)')


def snapshot_codebook_utilization(model, label: str, save_dir: str) -> None:
    """
    Compute and save a codebook utilization heatmap for a model at a given training stage.

    The model must implement `get_codebook_utilization() -> np.ndarray` returning a
    (n_digits, codebook_size) count array.  If the method is absent the call is a no-op,
    so this is safe to call for any model type.

    Args:
        model   : The (unwrapped) model instance.
        label   : Short string describing the snapshot stage, e.g. 'epoch_0' or 'final'.
        save_dir: Directory where the PNG file will be written.
    """
    if not hasattr(model, 'get_codebook_utilization'):
        return

    counts = model.get_codebook_utilization()
    model_name = model.__class__.__name__

    if hasattr(model, 'use_gumbel_noise'):
        noise_str = 'with_gumbel' if model.use_gumbel_noise else 'no_gumbel'
        title    = f'{model_name} ({noise_str}) — Codebook Utilization [{label}]'
        filename = f'{model_name}_{noise_str}_{label}.png'
    else:
        title    = f'{model_name} — Codebook Utilization [{label}]'
        filename = f'{model_name}_{label}.png'

    save_path = os.path.join(save_dir, filename)
    plot_codebook_heatmap(counts, title=title, save_path=save_path)
