import numpy as np
import polars as pl
import pandas as pd
import itertools
import seaborn as sns
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

# Function 1: Generating feature set and true labels
def generate_data_polars(N, p, seed=None):
    np.random.seed(seed)
    y = (np.random.rand(N) < p).astype(int)
    feat_1 = np.random.randint(0, 7, size=N)
    feat_2 = np.random.randint(0, 7, size=N)
    feat_3 = np.random.randint(0, 7, size=N)

    return pl.DataFrame({
        'feat_1': feat_1,
        'feat_2': feat_2,
        'feat_3': feat_3,
        'y': y
    })

# Function 2: Simulating ML predictions with confusion matrix
def simulate_ml_predictions_polars(y, alpha, beta, seed=None):
    np.random.seed(seed)
    y_np = y.to_numpy()
    y_hat = np.zeros_like(y_np)
    y_hat[y_np == 1] = (np.random.rand(np.sum(y_np == 1)) < alpha).astype(int)
    y_hat[y_np == 0] = (np.random.rand(np.sum(y_np == 0)) < beta).astype(int)
    return pl.Series('y_hat', y_hat)

# Function 3: Defining sampling probabilities
def define_sampling_polars(df):
    return df.with_columns([
        pl.when(pl.col('y_hat') == 1).then(0.8).otherwise(0.1).alias('ps')
    ])

# Function 4: Sampling data based on probabilities
def sample_data_polars(df, seed=None):
    np.random.seed(seed)
    rand = pl.Series('rand', np.random.rand(df.height))
    df = df.with_columns([rand])
    return df.with_columns([
        (pl.col('rand') < pl.col('ps')).alias('is_sampled')
    ])

# Function 5: Simulating human moderator labels
def simulate_moderator_polars(df, alpha_H, beta_H, seed=None):
    np.random.seed(seed)

    y = df['y'].to_numpy()
    is_sampled = df['is_sampled'].to_numpy()

    mod_label_vals = np.full(df.height, np.nan)  # use NaN instead of None

    sampled_y = y[is_sampled]
    mod_sampled = np.where(
        sampled_y == 1,
        (np.random.rand(len(sampled_y)) < alpha_H),
        (np.random.rand(len(sampled_y)) < beta_H)
    ).astype(int)

    mod_label_vals[is_sampled] = mod_sampled

    return df.with_columns(pl.Series("mod_label", mod_label_vals))

# Function 6: Estimating prevalence
def estimate_prevalence_polars(df, alpha_H, beta_H):
    sampled_df = df.filter(pl.col('is_sampled'))
    weights = 1 / sampled_df['ps']
    adjusted = (sampled_df['mod_label'] - beta_H) / (alpha_H - beta_H)
    p_hat = (adjusted * weights).sum() / df.height
    ci_width = 1.96 * adjusted.std() * weights.mean() / np.sqrt(sampled_df.height)
    return p_hat, (p_hat - ci_width, p_hat + ci_width)

# Grid of parameters
param_grid = list(itertools.product(
    [0.001, 0.01, 0.05, 0.1], # p
    [0.7, 0.85], # alpha
    [0.1, 0.2], # beta
    [0.85, 0.95], # alpha_H
    [0.05, 0.1] # beta_H
))

# Modified parameter grid
param_grid_base = list(itertools.product(
    [0.001, 0.01, 0.05, 0.1],          # p (true prevalence)
    [0.6, 0.7, 0.85, 0.9],             # alpha (model TPR)  (+ add 0.6, 0.9)
    [0.01, 0.05, 0.1, 0.2],            # beta  (model FPR)  (+ add lower-FPR cases)
    [0.75, 0.85, 0.95],                # alpha_H (human TPR) (+ add a weaker-human case)
    [0.05, 0.1, 0.2]                   # beta_H  (human FPR) (+ add non-ideal human case)
))

results_polars = []

def run_parameter_grid_simulation(N=100_000_0, param_grid=None, seed_offset=0, verbose=False):
    # default grid
    if param_grid is None:
        param_grid = list(itertools.product(
            [0.001, 0.01, 0.05, 0.1],   # p_true
            [0.7, 0.85],                # ML TPR (alpha)
            [0.1, 0.2],                 # ML FPR (beta)
            [0.85, 0.95],               # Moderator TPR (alpha_H)
            [0.05, 0.1]                 # Moderator FPR (beta_H)
        ))

    results = []

    # Main simulation loop
    for i, (p, alpha, beta, alpha_H, beta_H) in enumerate(param_grid):
        if verbose:
            print(f"Running combo {i+1}/{len(param_grid)}: "
                  f"p={p}, α={alpha}, β={beta}, α_H={alpha_H}, β_H={beta_H}")

        # --- Pipeline steps ---
        df = generate_data_polars(N=N, p=p, seed=seed_offset + i)
        y_hat = simulate_ml_predictions_polars(df["y"], alpha, beta, seed=seed_offset + i + 100)
        df = df.with_columns([y_hat])
        df = define_sampling_polars(df)
        df = sample_data_polars(df, seed=seed_offset + i + 200)
        df = simulate_moderator_polars(df, alpha_H, beta_H, seed=seed_offset + i + 300)
        p_hat, conf_int = estimate_prevalence_polars(df, alpha_H, beta_H)

        # --- Store results ---
        results.append({
            "p_true": p,
            "alpha": alpha,
            "beta": beta,
            "alpha_H": alpha_H,
            "beta_H": beta_H,
            "p_hat": p_hat,
            "conf_low": conf_int[0],
            "conf_high": conf_int[1],
            "bias": p_hat - p,
            "ci_width": conf_int[1] - conf_int[0]
        })

    # Convert to Polars DataFrame
    results_df = pl.DataFrame(results)

    return results_df


def estimate_total_variance(params, B=100):
    """
    Estimate total (stochastic) variance by repeating the full simulation B times.
    Each run uses a new seed.
    """
    p_hats = []
    for b in range(B):
        df = generate_data_polars(N=params["N"], p=params["p"], seed=params["seed_offset"])
        y_hat = simulate_ml_predictions_polars(
            df["y"], params["alpha"], params["beta"], seed=params["seed_offset"] + b + 100)
        df = df.with_columns([y_hat])
        df = define_sampling_polars(df)
        df = sample_data_polars(df, seed=params["seed_offset"] + b + 200)
        df = simulate_moderator_polars(df, params["alpha_H"], params["beta_H"], seed=params["seed_offset"] + b + 300)
        p_hat, _ = estimate_prevalence_polars(df, params["alpha_H"], params["beta_H"])
        p_hats.append(p_hat)
    return np.var(p_hats, ddof=1)


def estimate_stage_variance(params, stage, B=50):
    p_hats = []

    for b in range(B):
        # Always starting new population for each repetition
        df = generate_data_polars(N=params["N"], p=params["p"], seed=params["seed_offset"])

        # ML stage — resimulate if shaking ML, otherwise fixed
        if stage == "ml":
            y_hat = simulate_ml_predictions_polars(
                df["y"], params["alpha"], params["beta"], seed=params["seed_offset"] + 1000 + b)
        else:
            y_hat = simulate_ml_predictions_polars(
                df["y"], params["alpha"], params["beta"], seed=params["seed_offset"] + 100)  # fixed baseline
        df = df.with_columns([y_hat])
        df = define_sampling_polars(df)

        # Sampling stage — resimulate if shaking sampling
        if stage == "sampling":
            df = sample_data_polars(df, seed=params["seed_offset"] + 2000 + b)
        else:
            df = sample_data_polars(df, seed=params["seed_offset"] + 200)

        # Moderator stage — resimulate only if shaking moderator
        if stage == "moderator":
            df = simulate_moderator_polars(df, params["alpha_H"], params["beta_H"],
                                           seed=params["seed_offset"] + 3000 + b)
        else:
            df = simulate_moderator_polars(df, params["alpha_H"], params["beta_H"],
                                           seed=params["seed_offset"] + 300)
        # Estimate prevalence
        p_hat, _ = estimate_prevalence_polars(df, params["alpha_H"], params["beta_H"])
        p_hats.append(p_hat)

    # Handle degenerate cases
    p_hats = np.array(p_hats)
    if np.all(np.isnan(p_hats)) or len(np.unique(p_hats)) == 1:
        return np.nan
    return np.nanvar(p_hats, ddof=1)

def run_parameter_grid_with_variance(N=100_000,
                                     param_grid=None,
                                     B=200,
                                     seed_offset=0,
                                     verbose=False):
    """
    Run the prevalence simulation and estimate stochastic variance components
    (total and stage-wise) for each parameter setting.
    """
    # Reusing existing simulation for main results
    results_df = run_parameter_grid_simulation(N=N, param_grid=param_grid, seed_offset=seed_offset, verbose=verbose)
    results_pd = results_df.to_pandas()

    # Storage for variance diagnostics
    variance_records = []

    for i, row in results_pd.iterrows():
        params = dict(
            N=N,
            p=row["p_true"],
            alpha=row["alpha"],
            beta=row["beta"],
            alpha_H=row["alpha_H"],
            beta_H=row["beta_H"],
            seed_offset=seed_offset + i * 10_000
        )

        if verbose:
            print(f"→ Estimating variances for p={params['p']}, α={params['alpha']}, β={params['beta']}, "
                  f"α_H={params['alpha_H']}, β_H={params['beta_H']}")

        var_total = estimate_total_variance(params, B=B)
        var_ml = estimate_stage_variance(params, stage="ml", B=B//2) # Smaller B for faster run-time
        var_sampling = estimate_stage_variance(params, stage="sampling", B=B//2)
        var_mod = estimate_stage_variance(params, stage="moderator", B=B//2)
        var_inter = var_total - (var_ml + var_sampling + var_mod)

        variance_records.append({
            "p_true": params["p"],
            "alpha": params["alpha"],
            "beta": params["beta"],
            "alpha_H": params["alpha_H"],
            "beta_H": params["beta_H"],
            "var_stoch_total": var_total,
            "var_stoch_ml": var_ml,
            "var_stoch_sampling": var_sampling,
            "var_stoch_mod": var_mod,
            "var_stoch_inter": var_inter
        })

    variance_df = pl.DataFrame(variance_records)

    # Merge results and variances
    combined = results_df.join(variance_df, on=["p_true", "alpha", "beta", "alpha_H", "beta_H"])
    return combined


results_with_variance = run_parameter_grid_with_variance(
    N=50_000,  # smaller N for presentation
    param_grid=param_grid_base,
    B=30,      # repetitions per variance estimate
    verbose=True
)

print(results_with_variance.head())

results_with_variance_calc = results_with_variance.with_columns([
    (pl.col("var_stoch_ml") / pl.col("var_stoch_total")).alias("share_ml"),
    (pl.col("var_stoch_sampling") / pl.col("var_stoch_total")).alias("share_sampling"),
    (pl.col("var_stoch_mod") / pl.col("var_stoch_total")).alias("share_mod"),
])

df_calc = results_with_variance_calc.to_pandas()

df_plot = results_with_variance.to_pandas().rename(columns={
    "var_stoch_total": "Total stochastic variance",
    "p_true": "True prevalence $p$",
    "alpha_H": "Moderator TPR"
})

sns.barplot(
    data=df_plot,
    x="True prevalence $p$",
    y="Total stochastic variance",
    hue="Moderator TPR"
)
plt.title("Total stochastic variance by moderator accuracy")
plt.ylabel("Total stochastic variance")
plt.xlabel("True prevalence $p$")
plt.legend(title="Moderator TPR")
plt.show()


## Outside function env below
results_df_polars = run_parameter_grid_simulation(N=100_000, param_grid=param_grid_base, verbose=True)

#
print(results_df_polars)
print(len(results_df_polars))

## Graph: CI width by moderator TPR
results_pd = results_df_polars.to_pandas()

results_pd['ci_width'] = results_pd['conf_high'] - results_pd['conf_low']

sns.boxplot(data=results_pd, x='alpha_H', y='ci_width')
plt.title(r"Confidence Interval Width by Moderator TPR ($\alpha_H$)")
plt.ylabel("Confidence Interval Width")
plt.xlabel(r"Moderator TPR ($\alpha_H$)")
plt.show()


results_plot = results_with_variance.select([
    "p_true",
    "alpha",
    "beta",
    "alpha_H",
    "beta_H",
    "var_stoch_ml",
    "var_stoch_sampling",
    "var_stoch_mod"
])

df_melt = results_plot.to_pandas().melt(
    id_vars=["p_true", "alpha", "beta", "alpha_H", "beta_H"],
    value_vars=["var_stoch_ml", "var_stoch_sampling", "var_stoch_mod"],
    var_name="Stage",
    value_name="Variance"
)

stage_labels = {
    "var_stoch_ml": "ML",
    "var_stoch_sampling": "Sampling",
    "var_stoch_mod": "Moderator"
}

df_melt["Stage"] = df_melt["Stage"].map(stage_labels)

plt.figure(figsize=(8,5))
sns.barplot(
    data=df_melt,
    x="p_true",
    y="Variance",
    hue="Stage"
)
plt.title("Marginal Variance Contributions by Stage")
plt.ylabel("Variance")
plt.xlabel("True Prevalence rate $p$")
plt.legend(title="Variance Source")
plt.show()