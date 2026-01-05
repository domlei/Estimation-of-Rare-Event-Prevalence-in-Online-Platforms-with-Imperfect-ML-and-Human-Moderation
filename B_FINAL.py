import numpy as np
import polars as pl
import pandas as pd
import itertools
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Defined sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Risk-points definition
def compute_risk_points(f1, f2, f3, f4, f5):
    """
    f1..f5 are arrays in [0,1] representing:
    f1: account_age_norm (0=new,1=old)
    f2: activity_burstiness
    f3: link_intensity
    f4: text_effort_low
    f5: duplication_score
    """
    rp  = 2.0 * (1 - f1)  # newer → more risk
    rp += 2.5 * f2        # bursty → more risk
    rp += 2.0 * f3        # many links → more risk
    rp += 1.5 * f4        # low-effort text → more risk
    rp += 2.0 * f5        # duplication → more risk
    return rp

# Solver for b0 so mean(prob) ≈ target_prevalence
def solve_b0_for_target_prevalence_from_risk(risk_centered, target_prev,
                                             tol=1e-6, max_iter=50):
    """
    risk_centered: numpy array of risk_points - mean(risk_points)
    target_prev: desired average prevalence (e.g. 0.015 for 1.5%)
    """
    def prevalence_given_b0(b0):
        logits = b0 + risk_centered
        return sigmoid(logits).mean()

    low, high = -15.0, 5.0  # wide bracket in log-odds space

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        prev_mid = prevalence_given_b0(mid)

        if abs(prev_mid - target_prev) < tol:
            return mid

        if prev_mid < target_prev:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def generate_data_and_ml_B_polars(N, target_prev, alpha_target, beta_target, seed=None):
    """
    Approach B:
    - Generate features realistic for marketplace + social media
    - Generate true y via logistic model with intercept b0 tuned to target_prev
    - Fit logistic regression and threshold scores to approximate (alpha_target, beta_target)
    Returns:
      df_polars, metrics_dict
    """
    rng = np.random.default_rng(seed)

    # 1) Generate shared features
    # f1: account_age_norm (0=new,1=old)
    account_age_days = rng.exponential(scale=180, size=N)
    f1 = account_age_days / (account_age_days + 365)

    # f2: activity_burstiness
    actions_last_24h = rng.poisson(lam=3, size=N)
    f2 = np.clip(actions_last_24h / 30, 0, 1)

    # f3: link_intensity
    base_link_prop = rng.beta(1, 5, size=N)
    heavy_linkers = rng.random(N) < 0.1
    f3 = np.where(
        heavy_linkers,
        np.clip(base_link_prop + 0.6, 0, 1),
        base_link_prop,
    )

    # f4: text_effort_low
    f4 = rng.beta(2, 4, size=N)

    # f5: duplication_score
    template_users = rng.random(N) < 0.1
    f5 = np.where(
        template_users,
        rng.beta(5, 1, size=N),
        rng.beta(1, 5, size=N),
    )

    X = np.column_stack([f1, f2, f3, f4, f5])

    # 2) True risk model → p_true → y
    risk_points = compute_risk_points(f1, f2, f3, f4, f5)
    risk_centered = risk_points - risk_points.mean()

    b0 = solve_b0_for_target_prevalence_from_risk(
        risk_centered=risk_centered,
        target_prev=target_prev,
    )

    logits = b0 + risk_centered
    p_true = sigmoid(logits)

    y = (rng.random(N) < p_true).astype(int)

    # 3) Fitting logistic regression
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]

    # 4) Threshold search to approximate (alpha_target, beta_target)
    s = scores
    P = y.sum()
    Nn = N - P
    if P == 0 or Nn == 0:
        raise ValueError("Need both positives and negatives in y for ML metrics.")

    order = np.argsort(-s)
    s_desc = s[order]
    y_desc = y[order]

    tp_cum = np.cumsum(y_desc)
    fp_cum = np.cumsum(1 - y_desc)
    TPR = tp_cum / P
    FPR = fp_cum / Nn

    d2 = (TPR - alpha_target) ** 2 + (FPR - beta_target) ** 2
    k_best = int(np.argmin(d2))
    threshold = s_desc[k_best]

    y_hat = (scores >= threshold).astype(int)

    # Confusion matrix
    TP = int(((y_hat == 1) & (y == 1)).sum())
    FP = int(((y_hat == 1) & (y == 0)).sum())
    TN = int(((y_hat == 0) & (y == 0)).sum())
    FN = int(((y_hat == 0) & (y == 1)).sum())

    TPR_final = TP / P
    FPR_final = FP / Nn

    # Dataframe of features and risk points
    df = pl.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "risk_points": risk_points,
        "p_true": p_true,
        "y": y,
        "score": scores,
        "y_hat": y_hat,
    })

    metrics = {
        "b0": b0,
        "true_prevalence": float(y.mean()),
        "mean_p_true": float(p_true.mean()),
        "ml_TPR": TPR_final,
        "ml_FPR": FPR_final,
        "ml_threshold": float(threshold),
        "P": int(P),
        "N": int(N),
    }

    return df, metrics

def define_sampling_polars_score(df, target_rate=0.1, gamma=1.5,
                                 ps_min=0.01, ps_max=0.8):
    """
    Approach B-risk: sampling probabilities based on continuous ML scores.

    target_rate: desired average sampling fraction (e.g. 0.1 = 10% of items)
    gamma: how aggressively to focus on high-score items
    """
    scores = df["score"].to_numpy()
    # avoid zeros
    scores = np.clip(scores, 1e-6, 1 - 1e-6)

    # importance weights proportional to score^gamma
    w = scores ** gamma

    # scale so that mean(ps) ≈ target_rate
    c = target_rate * len(scores) / w.sum()
    ps = np.clip(c * w, ps_min, ps_max)

    return df.with_columns(pl.Series("ps", ps))


def sample_data_polars(df, seed=None):
    np.random.seed(seed)
    rand = pl.Series('rand', np.random.rand(df.height))
    df = df.with_columns([rand])
    return df.with_columns([
        (pl.col('rand') < pl.col('ps')).alias('is_sampled')
    ])

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

def estimate_prevalence_polars(df, alpha_H, beta_H):
    sampled_df = df.filter(pl.col('is_sampled'))
    weights = 1 / sampled_df['ps']
    adjusted = (sampled_df['mod_label'] - beta_H) / (alpha_H - beta_H)
    p_hat = (adjusted * weights).sum() / df.height
    ci_width = 1.96 * adjusted.std() * weights.mean() / np.sqrt(sampled_df.height)
    return p_hat, (p_hat - ci_width, p_hat + ci_width)

results_polars = []

def run_parameter_grid_simulation_B(N=100_000, param_grid=None, seed_offset=0, verbose=False):

    if param_grid is None:
        param_grid = list(itertools.product(
            [0.001, 0.01, 0.05, 0.1],   # target prevalence (p)
            [0.7, 0.85],                # target ML TPR (alpha)
            [0.1, 0.2],                 # target ML FPR (beta)
            [0.85, 0.95],               # Moderator TPR (alpha_H)
            [0.05, 0.1]                 # Moderator FPR (beta_H)
        ))

    results = []

    for i, (p, alpha, beta, alpha_H, beta_H) in enumerate(param_grid):
        if verbose:
            print(f"Running combo {i+1}/{len(param_grid)}: "
                  f"p_target={p}, α_target={alpha}, β_target={beta}, "
                  f"α_H={alpha_H}, β_H={beta_H}")

        # Approach B: data + ML predictions
        df, ml_metrics = generate_data_and_ml_B_polars(
            N=N,
            target_prev=p,
            alpha_target=alpha,
            beta_target=beta,
            seed=seed_offset + i,
        )

        # Sampling, moderator, estimator
        df = define_sampling_polars_score(df)
        df = sample_data_polars(df, seed=seed_offset + i + 200)
        df = simulate_moderator_polars(df, alpha_H, beta_H, seed=seed_offset + i + 300)
        p_hat, conf_int = estimate_prevalence_polars(df, alpha_H, beta_H)

        # realized prevalence from y
        realized_prev = float(df["y"].mean())

        # Store results + intermediate metrics
        results.append({
            # targets
            "p_target": p,
            "alpha_target": alpha,
            "beta_target": beta,
            "alpha_H": alpha_H,
            "beta_H": beta_H,

            # world + ML
            "true_prevalence": realized_prev,
            "mean_p_true": ml_metrics["mean_p_true"],
            "b0": ml_metrics["b0"],
            "ml_TPR": ml_metrics["ml_TPR"],
            "ml_FPR": ml_metrics["ml_FPR"],
            "ml_threshold": ml_metrics["ml_threshold"],

            # estimator
            "p_hat": p_hat,
            "conf_low": conf_int[0],
            "conf_high": conf_int[1],
            "bias_vs_target": p_hat - p,
            "bias_vs_realized": p_hat - realized_prev,
            "ci_width": conf_int[1] - conf_int[0],
        })

    results_df = pl.DataFrame(results)
    return results_df

p_list = [0.001, 0.01]                 #
alpha_ml_list = [0.85]                 # fix ML (or add 0.70 as another column)
beta_ml_list  = [0.10]                 # fix ML FPR

alpha_H_list = [0.70, 0.80, 0.85, 0.90, 0.95]   # moderator TPR
beta_H_list  = [0.01, 0.03, 0.05, 0.08, 0.12]   # moderator FPR

param_grid_biasmap_modq = list(itertools.product(
    p_list,
    alpha_ml_list,
    beta_ml_list,
    alpha_H_list,
    beta_H_list
))

# Deploying simulation
results_df_polars = run_parameter_grid_simulation_B(N=1_000_00, param_grid=param_grid_biasmap_modq,verbose=True)

print(results_df_polars)


pivot = (
    results_df_polars
    .to_pandas()
    .pivot_table(
        index="alpha_H",
        columns="beta_H",
        values="bias_vs_realized",
        aggfunc="median"
#        aggfunc="mean"
    )
)

print(pivot)

sns.heatmap(
    pivot,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Bias Heatmap by Moderator accuracy")
plt.xlabel(r"Moderator FPR ($\beta_H$)")
plt.ylabel(r"Moderator TPR ($\alpha_H$)")
plt.show()

###
# Graph: CI width by moderator TPR
results_pd = results_df_polars.to_pandas()

results_pd['ci_width'] = results_pd['conf_high'] - results_pd['conf_low']

sns.boxplot(data=results_pd, x='alpha_H', y='ci_width')
plt.title("Confidence Interval Width by Moderator accuracy")
plt.ylabel("Confidence Interval Width")
plt.xlabel(r"Moderator TPR ($\alpha_H$)")
plt.show()
###

# Graph: B bias heatmap - median

p_list = [0.001, 0.01]  # rare vs less rare

# Iterable
alpha_H_list = [0.90]   # moderator TPR (α_H)
beta_H_list  = [0.05]   # moderator FPR (β_H)

# ML target grid (axes of heatmap)
alpha_ml_list = [0.70, 0.80, 0.85, 0.90, 0.95]   # ML target TPR (α)
beta_ml_list  = [0.02, 0.05, 0.10, 0.15, 0.20]   # ML target FPR (β)

param_grid_biasmap = list(itertools.product(
    p_list,
    alpha_ml_list,
    beta_ml_list,
    alpha_H_list,
    beta_H_list
))

results_df_polars = run_parameter_grid_simulation_B(
    N=1_000_000,
    param_grid=param_grid_biasmap,
    verbose=True
)

results_pd = results_df_polars.to_pandas()


pivot_all = (
    results_pd
    .pivot_table(
        index="alpha_target",        # ML target TPR (α)
        columns="beta_target",       # ML target FPR (β)
        values="bias_vs_realized",
        aggfunc="median"
    )
    .sort_index()
    .sort_index(axis=1)
)

plt.figure(figsize=(6.8, 4.2))
sns.heatmap(pivot_all, annot=True, cmap="coolwarm", center=0)
plt.title("Bias heatmap over ML targets (median over all scenarios)")
plt.xlabel(r"ML target FPR ($\beta$)")
plt.ylabel(r"ML target TPR ($\alpha$)")
plt.tight_layout()
plt.show()