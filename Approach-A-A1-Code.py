import numpy as np
import polars as pl
import pandas as pd
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def generate_data_polars(N, p, seed=None):
    """
    Generating population data with true prevalence
    Only includes true label y.
    """
    rng = np.random.default_rng(seed)

    y = (rng.random(N) < p).astype(int)

    return pl.DataFrame({
        "id": np.arange(N),
        "y": y
    })

# A (SRS): constant sampling probability
def define_sampling_polars_A(df, ps_const=0.1):
    return df.with_columns([
        pl.lit(ps_const).alias("ps")
    ])

def sample_data_polars(df, seed=None):
    np.random.seed(seed)
    rand = pl.Series('rand', np.random.rand(df.height))
    df = df.with_columns([rand])
    return df.with_columns([(pl.col('rand') < pl.col('ps')).alias('is_sampled')])

def simulate_ml_predictions_polars(y: pl.Series, alpha, beta, seed=None):
    """
    Simulating binary ML predictions y_hat with:
      P(y_hat=1 | y=1) = alpha  (TPR)
      P(y_hat=1 | y=0) = beta   (FPR)
    """
    rng = np.random.default_rng(seed)

    y_np = y.to_numpy()
    u = rng.random(len(y_np))

    y_hat = np.where(
        y_np == 1,
        (u < alpha).astype(int),
        (u < beta).astype(int)
    )

    return pl.Series("y_hat", y_hat)

# Simulating inclusion for moderation based on selected rate
def simulate_moderator_polars(df, alpha_H, beta_H, seed=None):
    np.random.seed(seed)
    y = df['y'].to_numpy()
    is_sampled = df['is_sampled'].to_numpy()
    mod_label_vals = np.full(df.height, np.nan)
    sampled_y = y[is_sampled]
    mod_sampled = np.where(
        sampled_y == 1,
        (np.random.rand(len(sampled_y)) < alpha_H),
        (np.random.rand(len(sampled_y)) < beta_H)
    ).astype(int)
    mod_label_vals[is_sampled] = mod_sampled
    return df.with_columns(pl.Series("mod_label", mod_label_vals))


# A1: defining sampling probability
def define_sampling_polars_A1(df, ps_pos=0.8, ps_neg=0.1):
    return df.with_columns([
        pl.when(pl.col("y_hat") == 1).then(ps_pos).otherwise(ps_neg).alias("ps")
    ])

def estimate_prevalence_polars(df, alpha_H, beta_H):
    sampled_df = df.filter(pl.col('is_sampled'))
    weights = 1 / sampled_df['ps']
    adjusted = (sampled_df['mod_label'] - beta_H) / (alpha_H - beta_H)
    p_hat = (adjusted * weights).sum() / df.height
    ci_width = 1.96 * adjusted.std() * weights.mean() / np.sqrt(max(sampled_df.height, 1))
    return p_hat, (p_hat - ci_width, p_hat + ci_width)

# Monte Carlo runner for A vs A1 (proper bias/var/mse across R runs)
def run_A_vs_A1_mc(
    N=1_000_000,
    p_grid=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
    alpha=0.85, beta=0.10,
    alpha_H=0.90, beta_H=0.05,

    # baseline A design grid
    a_grid=(0.05, 0.10),

    # A1 design grid remains the same
    a1_grid=((0.60, 0.05), (0.80, 0.05), (0.95, 0.05), (0.95, 0.20)),
    R=200,
    seed_offset=0
):
    records = []

    for p in p_grid:

        # Approach A (SRS)
        for ps_const in a_grid:
            p_hats_A = []
            for r in range(R):
                seed = seed_offset + 10_000 * r + int(1e6 * p)

                df = generate_data_polars(N=N, p=p, seed=seed)
                y_hat = simulate_ml_predictions_polars(df["y"], alpha, beta, seed=seed + 100)
                df = df.with_columns([y_hat])

                df = define_sampling_polars_A(df, ps_const=ps_const)
                df = sample_data_polars(df, seed=seed + 200)
                df = simulate_moderator_polars(df, alpha_H, beta_H, seed=seed + 300)

                p_hat, _ = estimate_prevalence_polars(df, alpha_H, beta_H)
                p_hats_A.append(p_hat)

            p_hats_A = np.array(p_hats_A)
            err_A = p_hats_A - p

            records.append({
                "approach": "A",
                "p_true": p,

                # design parameters
                "ps_const": ps_const,
                "ps_pos": None,
                "ps_neg": None,

                # metrics
                "bias": float(err_A.mean()),
                "var": float(p_hats_A.var(ddof=1)),
                "mse": float((err_A**2).mean()),
                "rmse": float(np.sqrt((err_A**2).mean())),
                "mean_p_hat": float(p_hats_A.mean())
            })

        # Approach A1 (unequal-probabilities)
        for (ps_pos, ps_neg) in a1_grid:
            p_hats_A1 = []
            for r in range(R):
                seed = seed_offset + 10_000 * r + int(1e6 * p) + 777

                df = generate_data_polars(N=N, p=p, seed=seed)
                y_hat = simulate_ml_predictions_polars(df["y"], alpha, beta, seed=seed + 100)
                df = df.with_columns([y_hat])

                df = define_sampling_polars_A1(df, ps_pos=ps_pos, ps_neg=ps_neg)
                df = sample_data_polars(df, seed=seed + 200)
                df = simulate_moderator_polars(df, alpha_H, beta_H, seed=seed + 300)

                p_hat, _ = estimate_prevalence_polars(df, alpha_H, beta_H)
                p_hats_A1.append(p_hat)

            p_hats_A1 = np.array(p_hats_A1)
            err_A1 = p_hats_A1 - p

            records.append({
                "approach": "A1",
                "p_true": p,

                # design parameters
                "ps_const": None,
                "ps_pos": ps_pos,
                "ps_neg": ps_neg,

                # metrics
                "bias": float(err_A1.mean()),
                "var": float(p_hats_A1.var(ddof=1)),
                "mse": float((err_A1**2).mean()),
                "rmse": float(np.sqrt((err_A1**2).mean())),
                "mean_p_hat": float(p_hats_A1.mean())
            })

    return pl.DataFrame(records)


# Plotting functions

def plot_rmse_vs_p(df: pl.DataFrame):
    pdf = df.to_pandas()

    # Creating presentable labels
    def policy_label(row):
        if row["approach"] == "A":
            return rf"A ($p_s={row['ps_const']:.2f}$)"
        return rf"A1 ($p_s(1)={row['ps_pos']:.2f},\; p_s(0)={row['ps_neg']:.2f}$)"

    pdf["label"] = pdf.apply(policy_label, axis=1)

    plt.figure()
    for label, sub in pdf.groupby("label"):
        sub = sub.sort_values("p_true")
        plt.plot(sub["p_true"], sub["rmse"], marker="o", label=label)

    plt.xscale("log")
    plt.xlabel(r"True prevalence ($p$) (log scale)")
    plt.ylabel("RMSE of estimated prevalence")
    plt.title("A vs A1: Sampling design effect (RMSE)")
    plt.legend()
    plt.tight_layout()
    plt.show()


df_mc = run_A_vs_A1_mc(
    N=1_000_000,
    p_grid=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
    alpha=0.85, beta=0.10,
    alpha_H=0.90, beta_H=0.05,
    a_grid=(0.05, 0.10),
    a1_grid=((0.60, 0.05), (0.80, 0.05), (0.95, 0.05), (0.95, 0.20)),
    R=200,
    seed_offset=123
)
print(df_mc)

plot_rmse_vs_p(df_mc)

def plot_bias2_vs_var_at_p(df_mc, p0=0.005):
    """
    At one prevalence p0, compare Bias^2 vs Variance across sampling policies (A and A1 variants).
    columns: approach, p_true, ps_const, ps_pos, ps_neg, bias, var
    """
    pdf = df_mc.to_pandas()
    sub = pdf[np.isclose(pdf["p_true"], p0)].copy()

    # Creating presentable labels
    def label_row(r):
        if r["approach"] == "A":
            return rf"A (SRS, $p_s={r['ps_const']:.2f}$)"
        return rf"A1 ($p_s(1)={r['ps_pos']:.2f},\; p_s(0)={r['ps_neg']:.2f}$)"

    sub["label"] = sub.apply(label_row, axis=1)

    # Metrics
    sub["bias2"] = sub["bias"] ** 2
    sub = sub.sort_values(["approach", "ps_pos", "ps_neg"], na_position="first")

    labels = sub["label"].tolist()
    x = np.arange(len(labels))
    width = 0.40

    plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, sub["var"].to_numpy(), width, label="Variance")
    plt.bar(x + width/2, sub["bias2"].to_numpy(), width, label=r"Bias$^2$")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=11)
    plt.ylabel("Contribution to MSE", fontsize=12)
    plt.title(rf"Bias-variance relationship at $p={p0:g}$", fontsize=13)

    plt.legend(fontsize=10, title_fontsize=11)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


# Deploy:
plot_bias2_vs_var_at_p(df_mc, p0=0.005)
#plot_bias2_vs_var_at_p(df_mc, p0=0.01)


def plot_lollipop_bias(
    df_mc,
    approach="A",
    title="Bias vs true prevalence (lollipop plot)",
    log_x=False,
    color_map=None
):
    """
    Lollipop (pole) plot of bias vs prevalence, with colors separating sampling regimes.

    Parameters
    ----------
    df_mc : DataFrame (polars or pandas)
        Columns: approach, p_true, bias
        If approach == "A", should also contain ps_const for regime coloring.
    approach : str
        Which approach to plot (default: "A" for SRS)
    title : str
        Plot title
    log_x : bool
        Whether to use log scale for x-axis (default: False)
    color_map : dict or None
        Optional mapping {ps_const_value: matplotlib_color}. If None, uses defaults.
    """
    # Convert to pandas if needed
    if hasattr(df_mc, "to_pandas"):
        pdf = df_mc.to_pandas()
    else:
        pdf = df_mc.copy()

    # Filter to the requested approach only
    sub = pdf[pdf["approach"] == approach].copy()
    sub = sub.sort_values("p_true")

    plt.figure(figsize=(6, 3.5))

    # Default colors for baseline regimes (override via color_map if you want)
    if color_map is None:
        color_map = {
            0.05: "tab:blue",
            0.10: "tab:orange",
        }

    # If ps_const exists, color by sampling regime; otherwise fall back to single-color plot
    if "ps_const" in sub.columns and sub["ps_const"].notna().any():
        for ps_val, g in sub.groupby("ps_const"):
            g = g.sort_values("p_true")
            c = color_map.get(ps_val, "gray")

            plt.vlines(
                g["p_true"],
                0,
                g["bias"],
                color=c,
                alpha=0.7,
                linewidth=2
            )
            plt.scatter(
                g["p_true"],
                g["bias"],
                color=c,
                label=fr"SRS ($\pi$ = {ps_val:.2f})",
                zorder=3
            )
        plt.legend(frameon=True)
    else:
        # Generic single-series lollipop plot
        plt.vlines(sub["p_true"], 0, sub["bias"], alpha=0.7, linewidth=2)
        plt.scatter(sub["p_true"], sub["bias"], zorder=3)

    plt.axhline(0, color="black", linewidth=1)

    if log_x:
        plt.xscale("log")
        plt.xlabel(r"True prevalence $p$ (log scale)")
    else:
        plt.xlabel(r"True prevalence $p$")

    plt.ylabel(r"Bias $\mathbb{E}[\hat p - p]$")
    plt.title(title)

    plt.tight_layout()
    plt.show()


plot_lollipop_bias(
    df_mc,
    approach="A",
    title="Baseline (SRS): Bias vs prevalence"

)
