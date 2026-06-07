from __future__ import annotations

import argparse
import itertools
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


DPS_DIMS = ["I", "F", "P", "C", "S", "N", "FA"]
PROFILE_ORDER = ["low_payment_browsing", "high_payment_shopping", "bargain_hunter"]
PROFILE_LABELS = {
    "low_payment_browsing": "Low Payment",
    "high_payment_shopping": "High Payment",
    "bargain_hunter": "Bargain Hunter",
}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), used for chi-square p-values."""
    if a <= 0:
        return float("nan")
    if x < 0:
        return float("nan")
    if x == 0:
        return 1.0

    eps = 3.0e-14
    max_iter = 1000
    gln = math.lgamma(a)

    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        total = delta
        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                lower_p = total * math.exp(-x + a * math.log(x) - gln)
                return max(0.0, min(1.0, 1.0 - lower_p))
        lower_p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - lower_p))

    b = x + 1.0 - a
    c = 1.0 / 1.0e-300
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-300:
            d = 1.0e-300
        c = b + an / c
        if abs(c) < 1.0e-300:
            c = 1.0e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            upper_q = math.exp(-x + a * math.log(x) - gln) * h
            return max(0.0, min(1.0, upper_q))
    upper_q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, upper_q))


def chi2_sf(x: float, dof: int) -> float:
    if dof <= 0:
        return float("nan")
    return gammaincc(dof / 2.0, x / 2.0)


def betacf(a: float, b: float, x: float) -> float:
    eps = 3.0e-14
    fpmin = 1.0e-300
    max_iter = 200

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, dof: float) -> float:
    if not math.isfinite(t_value) or dof <= 0:
        return float("nan")
    x = dof / (dof + t_value * t_value)
    ib = regularized_beta(x, dof / 2.0, 0.5)
    if t_value >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def p_to_stars(p_value: float, alpha: float) -> str:
    if not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return ""


def ordered_profiles(values: Iterable[str]) -> List[str]:
    seen = set(values)
    ordered = [p for p in PROFILE_ORDER if p in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def display_profile(profile_id: str) -> str:
    return PROFILE_LABELS.get(profile_id, profile_id)


def load_tables(all_steps: Path, by_type_dir: Optional[Path]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if all_steps.exists():
        tables["all"] = pd.read_csv(all_steps, encoding="utf-8-sig")
    if by_type_dir and by_type_dir.exists():
        for path in sorted(by_type_dir.glob("all_steps_*.csv")):
            scope = path.stem.replace("all_steps_", "", 1)
            tables[scope] = pd.read_csv(path, encoding="utf-8-sig")

    missing = {}
    required = {"profile_id", *DPS_DIMS}
    for scope, df in tables.items():
        absent = required - set(df.columns)
        if absent:
            missing[scope] = sorted(absent)
    if missing:
        details = "; ".join(f"{scope}: {cols}" for scope, cols in missing.items())
        raise ValueError(f"Missing required columns: {details}")
    if not tables:
        raise FileNotFoundError("No input tables found.")
    return tables


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for dim in DPS_DIMS:
        cleaned[dim] = pd.to_numeric(cleaned[dim], errors="coerce").fillna(0.0)
    cleaned["profile_id"] = cleaned["profile_id"].astype(str)
    if "app" in cleaned.columns:
        cleaned["app"] = cleaned["app"].astype(str)
    return cleaned


def exposure_summary(scope: str, df: pd.DataFrame) -> List[dict]:
    rows = []
    group_cols = ["profile_id"]
    if "app" in df.columns:
        group_cols.append("app")

    for keys, sub_df in df.groupby(group_cols, dropna=False):
        if isinstance(keys, tuple):
            profile_id = keys[0]
            app = keys[1]
        else:
            profile_id = keys
            app = "all"
        for dim in DPS_DIMS:
            values = sub_df[dim].astype(float)
            rows.append(
                {
                    "scope": scope,
                    "app": app,
                    "profile_id": profile_id,
                    "profile_label": display_profile(profile_id),
                    "dimension": dim,
                    "n_steps": int(len(values)),
                    "total_score": float(values.sum()),
                    "mean_score": float(values.mean()) if len(values) else float("nan"),
                    "sd_score": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "nonzero_steps": int((values > 0).sum()),
                    "nonzero_rate": float((values > 0).mean()) if len(values) else float("nan"),
                }
            )

    for profile_id, sub_df in df.groupby("profile_id", dropna=False):
        for dim in DPS_DIMS:
            values = sub_df[dim].astype(float)
            rows.append(
                {
                    "scope": scope,
                    "app": "ALL_APPS",
                    "profile_id": profile_id,
                    "profile_label": display_profile(profile_id),
                    "dimension": dim,
                    "n_steps": int(len(values)),
                    "total_score": float(values.sum()),
                    "mean_score": float(values.mean()) if len(values) else float("nan"),
                    "sd_score": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "nonzero_steps": int((values > 0).sum()),
                    "nonzero_rate": float((values > 0).mean()) if len(values) else float("nan"),
                }
            )
    return rows


def contingency_table(df: pd.DataFrame, count_mode: str) -> pd.DataFrame:
    values = df[DPS_DIMS].astype(float)
    if count_mode == "binary":
        values = (values > 0).astype(float)
    table = values.groupby(df["profile_id"]).sum()
    table = table.reindex(index=ordered_profiles(table.index), columns=DPS_DIMS).fillna(0.0)
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    return table


def chi_square_from_observed(observed_df: pd.DataFrame) -> Tuple[float, int, float, np.ndarray]:
    observed = observed_df.to_numpy(dtype=float)
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return float("nan"), 0, float("nan"), np.full_like(observed, np.nan)

    if scipy_stats is not None:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed, correction=False)
        return float(chi2), int(dof), float(p_value), expected

    total = observed.sum()
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / total if total > 0 else np.zeros_like(observed)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    chi2 = float(chi2_terms.sum())
    dof = int((observed.shape[0] - 1) * (observed.shape[1] - 1))
    return chi2, dof, chi2_sf(chi2, dof), expected


def cramers_v(chi2: float, observed_df: pd.DataFrame) -> float:
    observed = observed_df.to_numpy(dtype=float)
    n = observed.sum()
    k = min(observed.shape)
    if n <= 0 or k <= 1 or not math.isfinite(chi2):
        return float("nan")
    return math.sqrt(max(0.0, chi2 / (n * (k - 1))))


def chi_square_rows(scope: str, df: pd.DataFrame, count_mode: str, alpha: float) -> Tuple[List[dict], pd.DataFrame]:
    rows = []
    global_table = contingency_table(df, count_mode)
    chi2, dof, p_value, expected = chi_square_from_observed(global_table)
    min_expected = float(np.nanmin(expected)) if expected.size else float("nan")
    low_expected_cells = int(np.sum(expected < 5)) if expected.size else 0
    rows.append(
        {
            "scope": scope,
            "comparison": "global",
            "profile_a": "ALL_PROFILES",
            "profile_b": "",
            "profiles": "|".join(global_table.index.astype(str)),
            "count_mode": count_mode,
            "n_total": float(global_table.to_numpy(dtype=float).sum()),
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
            "cramers_v": cramers_v(chi2, global_table),
            "min_expected": min_expected,
            "low_expected_cells": low_expected_cells,
            "significant": bool(math.isfinite(p_value) and p_value < alpha),
            "stars": p_to_stars(p_value, alpha),
        }
    )

    profiles = ordered_profiles(df["profile_id"].unique())
    for profile_a, profile_b in itertools.combinations(profiles, 2):
        pair_df = df[df["profile_id"].isin([profile_a, profile_b])]
        table = contingency_table(pair_df, count_mode)
        chi2, dof, p_value, expected = chi_square_from_observed(table)
        min_expected = float(np.nanmin(expected)) if expected.size else float("nan")
        low_expected_cells = int(np.sum(expected < 5)) if expected.size else 0
        rows.append(
            {
                "scope": scope,
                "comparison": "pairwise",
                "profile_a": profile_a,
                "profile_b": profile_b,
                "profiles": f"{profile_a}|{profile_b}",
                "count_mode": count_mode,
                "n_total": float(table.to_numpy(dtype=float).sum()),
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
                "cramers_v": cramers_v(chi2, table),
                "min_expected": min_expected,
                "low_expected_cells": low_expected_cells,
                "significant": bool(math.isfinite(p_value) and p_value < alpha),
                "stars": p_to_stars(p_value, alpha),
            }
        )
    return rows, global_table


def permutation_test_l1(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    mean_a = values_a.mean(axis=0)
    mean_b = values_b.mean(axis=0)
    observed_l1 = float(np.abs(mean_a - mean_b).sum())
    pooled = np.vstack([values_a, values_b])
    n_a = len(values_a)
    n_total = len(pooled)
    extreme = 0
    permuted = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        order = rng.permutation(n_total)
        perm_a = pooled[order[:n_a]]
        perm_b = pooled[order[n_a:]]
        delta = float(np.abs(perm_a.mean(axis=0) - perm_b.mean(axis=0)).sum())
        permuted[i] = delta
        if delta >= observed_l1 - 1.0e-12:
            extreme += 1

    p_value = (extreme + 1.0) / (n_permutations + 1.0)
    return (
        observed_l1,
        p_value,
        float(np.mean(permuted)),
        float(np.quantile(permuted, 0.95)),
        float(np.quantile(permuted, 0.99)),
    )


def permutation_rows(scope: str, df: pd.DataFrame, n_permutations: int, seed: int, alpha: float) -> List[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    profiles = ordered_profiles(df["profile_id"].unique())
    for profile_a, profile_b in itertools.combinations(profiles, 2):
        values_a = df[df["profile_id"] == profile_a][DPS_DIMS].to_numpy(dtype=float)
        values_b = df[df["profile_id"] == profile_b][DPS_DIMS].to_numpy(dtype=float)
        if len(values_a) == 0 or len(values_b) == 0:
            continue
        observed_l1, p_value, perm_mean, perm_q95, perm_q99 = permutation_test_l1(
            values_a, values_b, n_permutations, rng
        )
        rows.append(
            {
                "scope": scope,
                "profile_a": profile_a,
                "profile_b": profile_b,
                "n_a": int(len(values_a)),
                "n_b": int(len(values_b)),
                "n_permutations": int(n_permutations),
                "seed": int(seed),
                "distance": "L1",
                "observed_l1": observed_l1,
                "perm_mean_l1": perm_mean,
                "perm_q95_l1": perm_q95,
                "perm_q99_l1": perm_q99,
                "p_value": p_value,
                "significant": bool(p_value < alpha),
                "stars": p_to_stars(p_value, alpha),
            }
        )
    return rows


def welch_t_test(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float, float]:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan"), float("nan"), float("nan")
    if scipy_stats is not None:
        res = scipy_stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
        dof = welch_dof(values_a, values_b)
        return float(res.statistic), float(dof), float(res.pvalue)

    mean_a = float(values_a.mean())
    mean_b = float(values_b.mean())
    var_a = float(values_a.var(ddof=1))
    var_b = float(values_b.var(ddof=1))
    se = math.sqrt(var_a / len(values_a) + var_b / len(values_b))
    if se == 0:
        return 0.0, float("inf"), 1.0
    t_stat = (mean_a - mean_b) / se
    dof = welch_dof(values_a, values_b)
    cdf = student_t_cdf(t_stat, dof)
    p_value = 2.0 * min(cdf, 1.0 - cdf)
    return float(t_stat), float(dof), max(0.0, min(1.0, p_value))


def welch_dof(values_a: np.ndarray, values_b: np.ndarray) -> float:
    var_a = float(values_a.var(ddof=1))
    var_b = float(values_b.var(ddof=1))
    term_a = var_a / len(values_a)
    term_b = var_b / len(values_b)
    numerator = (term_a + term_b) ** 2
    denominator = 0.0
    if len(values_a) > 1:
        denominator += term_a**2 / (len(values_a) - 1)
    if len(values_b) > 1:
        denominator += term_b**2 / (len(values_b) - 1)
    return numerator / denominator if denominator > 0 else float("inf")


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def mann_whitney_u_test(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float, float]:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    n_a = len(values_a)
    n_b = len(values_b)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan"), float("nan")
    if scipy_stats is not None:
        res = scipy_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        z_stat = normal_quantile_from_p(float(res.pvalue))
        return float(res.statistic), z_stat, float(res.pvalue)

    pooled = np.concatenate([values_a, values_b])
    ranks = average_ranks(pooled)
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    mean_u = n_a * n_b / 2.0

    _, tie_counts = np.unique(pooled, return_counts=True)
    n_total = n_a + n_b
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    var_u = n_a * n_b / 12.0
    if n_total > 1:
        var_u *= (n_total + 1.0) - tie_term / (n_total * (n_total - 1.0))
    if var_u <= 0:
        return float(u_a), 0.0, 1.0
    z_abs = max(0.0, (abs(u_a - mean_u) - 0.5) / math.sqrt(var_u))
    p_value = 2.0 * (1.0 - normal_cdf(z_abs))
    z_stat = math.copysign(z_abs, u_a - mean_u)
    return float(u_a), float(z_stat), max(0.0, min(1.0, p_value))


def normal_quantile_from_p(p_value: float) -> float:
    if not math.isfinite(p_value) or p_value <= 0 or p_value >= 1:
        return float("nan")
    # Only used as a SciPy-path display field; leave as NaN rather than add another dependency.
    return float("nan")


def cohen_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    var_a = float(values_a.var(ddof=1))
    var_b = float(values_b.var(ddof=1))
    pooled_var = ((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / (
        len(values_a) + len(values_b) - 2
    )
    if pooled_var <= 0:
        return 0.0
    return float((values_a.mean() - values_b.mean()) / math.sqrt(pooled_var))


def dimension_test_rows(scope: str, df: pd.DataFrame, alpha: float) -> List[dict]:
    rows = []
    profiles = ordered_profiles(df["profile_id"].unique())
    for profile_a, profile_b in itertools.combinations(profiles, 2):
        sub_a = df[df["profile_id"] == profile_a]
        sub_b = df[df["profile_id"] == profile_b]
        for dim in DPS_DIMS:
            values_a = sub_a[dim].to_numpy(dtype=float)
            values_b = sub_b[dim].to_numpy(dtype=float)
            t_stat, t_dof, t_p = welch_t_test(values_a, values_b)
            u_stat, u_z, u_p = mann_whitney_u_test(values_a, values_b)
            mean_a = float(np.mean(values_a)) if len(values_a) else float("nan")
            mean_b = float(np.mean(values_b)) if len(values_b) else float("nan")
            rows.append(
                {
                    "scope": scope,
                    "profile_a": profile_a,
                    "profile_b": profile_b,
                    "dimension": dim,
                    "n_a": int(len(values_a)),
                    "n_b": int(len(values_b)),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "diff_a_minus_b": mean_a - mean_b,
                    "cohen_d": cohen_d(values_a, values_b),
                    "welch_t": t_stat,
                    "welch_dof": t_dof,
                    "welch_p_value": t_p,
                    "welch_significant": bool(math.isfinite(t_p) and t_p < alpha),
                    "mannwhitney_u": u_stat,
                    "mannwhitney_z": u_z,
                    "mannwhitney_p_value": u_p,
                    "mannwhitney_significant": bool(math.isfinite(u_p) and u_p < alpha),
                }
            )
    return rows


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    valid = p_values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return q_values
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def add_fdr_columns(df: pd.DataFrame, p_columns: List[str], alpha: float, group_cols: List[str]) -> pd.DataFrame:
    result = df.copy()
    group_by = group_cols[0] if len(group_cols) == 1 else group_cols
    for p_col in p_columns:
        q_col = p_col.replace("p_value", "q_value")
        result[q_col] = np.nan
        for _, idx in result.groupby(group_by, dropna=False).groups.items():
            result.loc[idx, q_col] = benjamini_hochberg(result.loc[idx, p_col])
        result[q_col.replace("q_value", "significant_fdr")] = result[q_col] < alpha
    return result


def save_contingency_tables(output_dir: Path, scope: str, table: pd.DataFrame) -> None:
    path = output_dir / f"contingency_{scope}.csv"
    table.to_csv(path, encoding="utf-8-sig")


def plot_scope_heatmap(scope: str, df: pd.DataFrame, output_dir: Path) -> None:
    mean_matrix = df.groupby("profile_id")[DPS_DIMS].mean()
    mean_matrix = mean_matrix.reindex(ordered_profiles(mean_matrix.index))
    mean_matrix.index = [display_profile(idx) for idx in mean_matrix.index]

    plt.figure(figsize=(9, 4.6))
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(1.0, float(np.nanmax(mean_matrix.to_numpy()))),
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Mean DPS exposure"},
    )
    plt.title(f"Mean E(D | A) by Profile - {scope}", fontsize=18, fontweight="bold")
    plt.xlabel("DPS dimension", fontsize=14, fontweight="bold")
    plt.ylabel("Profile", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_mean_exposure_{scope}.png", dpi=220)
    plt.close()


def plot_chi_square(chi_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = chi_df.copy()
    plot_df["label"] = np.where(
        plot_df["comparison"] == "global",
        "Global",
        plot_df["profile_a"].map(display_profile) + " vs " + plot_df["profile_b"].map(display_profile),
    )
    plt.figure(figsize=(12, 5.5))
    sns.barplot(data=plot_df, x="scope", y="cramers_v", hue="label")
    plt.axhline(0.1, color="#777777", linestyle="--", linewidth=1, label="V=0.10")
    plt.title("Cramer's V for Profile-Dark Pattern Association", fontsize=18, fontweight="bold")
    plt.xlabel("Scope", fontsize=14, fontweight="bold")
    plt.ylabel("Cramer's V", fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "cramers_v_by_scope.png", dpi=220)
    plt.close()


def plot_permutation(perm_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = perm_df.copy()
    plot_df["comparison"] = plot_df["profile_a"].map(display_profile) + " vs " + plot_df["profile_b"].map(display_profile)
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=plot_df, x="scope", y="observed_l1", hue="comparison")
    plt.title("Permutation Test Observed L1 Exposure Gap", fontsize=18, fontweight="bold")
    plt.xlabel("Scope", fontsize=14, fontweight="bold")
    plt.ylabel("Observed L1 distance", fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_observed_l1_by_scope.png", dpi=220)
    plt.close()

    plot_df["minus_log10_p"] = -np.log10(plot_df["p_value"].clip(lower=1.0e-300))
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=plot_df, x="scope", y="minus_log10_p", hue="comparison")
    plt.axhline(-math.log10(0.05), color="#777777", linestyle="--", linewidth=1, label="p=0.05")
    plt.title("Permutation Test Significance", fontsize=18, fontweight="bold")
    plt.xlabel("Scope", fontsize=14, fontweight="bold")
    plt.ylabel("-lg(p-value)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_pvalues_by_scope.png", dpi=220)
    plt.close()


def plot_dimension_tests(dim_df: pd.DataFrame, output_dir: Path) -> None:
    if "mannwhitney_q_value" not in dim_df.columns:
        return
    for scope, sub_df in dim_df.groupby("scope"):
        plot_df = sub_df.copy()
        plot_df["comparison"] = (
            plot_df["profile_a"].map(display_profile) + " vs " + plot_df["profile_b"].map(display_profile)
        )
        matrix = plot_df.pivot_table(
            index="comparison",
            columns="dimension",
            values="mannwhitney_q_value",
            aggfunc="min",
        ).reindex(columns=DPS_DIMS)
        if matrix.empty:
            continue
        transformed = -np.log10(matrix.clip(lower=1.0e-300))
        plt.figure(figsize=(9, max(3.8, 0.8 * len(matrix))))
        sns.heatmap(
            transformed,
            annot=matrix,
            fmt=".3f",
            cmap="Blues",
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "-lg(FDR q-value)"},
        )
        plt.title(f"Mann-Whitney U FDR q-values - {scope}", fontsize=18, fontweight="bold")
        plt.xlabel("DPS dimension", fontsize=14, fontweight="bold")
        plt.ylabel("Profile comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"mannwhitney_qvalue_heatmap_{scope}.png", dpi=220)
        plt.close()


def make_plots(tables: Dict[str, pd.DataFrame], chi_df: pd.DataFrame, perm_df: pd.DataFrame, dim_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

    for scope, df in tables.items():
        plot_scope_heatmap(scope, df, plot_dir)
    if not chi_df.empty:
        plot_chi_square(chi_df, plot_dir)
    if not perm_df.empty:
        plot_permutation(perm_df, plot_dir)
    if not dim_df.empty:
        plot_dimension_tests(dim_df, plot_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistical tests for dark-pattern exposure differences across profiles."
    )
    parser.add_argument("--all-steps", type=Path, default=Path("results/statistics/all_steps.csv"))
    parser.add_argument("--by-type-dir", type=Path, default=Path("results/statistics_by_type"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistical_tests"))
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--count-mode",
        choices=["binary", "score"],
        default="binary",
        help="Chi-square count table mode: binary counts D_i>0 occurrences; score sums raw DPS exposure mass.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_tables = load_tables(args.all_steps, args.by_type_dir)
    tables = {scope: clean_table(df) for scope, df in raw_tables.items()}

    summary_rows: List[dict] = []
    chi_rows_all: List[dict] = []
    permutation_rows_all: List[dict] = []
    dimension_rows_all: List[dict] = []

    for scope, df in tables.items():
        summary_rows.extend(exposure_summary(scope, df))
        chi_rows, contingency = chi_square_rows(scope, df, args.count_mode, args.alpha)
        chi_rows_all.extend(chi_rows)
        save_contingency_tables(args.output_dir, scope, contingency)
        permutation_rows_all.extend(permutation_rows(scope, df, args.permutations, args.seed, args.alpha))
        dimension_rows_all.extend(dimension_test_rows(scope, df, args.alpha))

    summary_df = pd.DataFrame(summary_rows)
    chi_df = pd.DataFrame(chi_rows_all)
    perm_df = pd.DataFrame(permutation_rows_all)
    dim_df = pd.DataFrame(dimension_rows_all)

    if not chi_df.empty:
        chi_df = add_fdr_columns(chi_df, ["p_value"], args.alpha, ["comparison"])
    if not perm_df.empty:
        perm_df = add_fdr_columns(perm_df, ["p_value"], args.alpha, ["scope"])
    if not dim_df.empty:
        dim_df = add_fdr_columns(
            dim_df,
            ["welch_p_value", "mannwhitney_p_value"],
            args.alpha,
            ["scope", "profile_a", "profile_b"],
        )

    summary_df.to_csv(args.output_dir / "exposure_summary.csv", index=False, encoding="utf-8-sig")
    chi_df.to_csv(args.output_dir / "chi_square_tests.csv", index=False, encoding="utf-8-sig")
    perm_df.to_csv(args.output_dir / "permutation_tests.csv", index=False, encoding="utf-8-sig")
    dim_df.to_csv(args.output_dir / "dimension_mean_tests.csv", index=False, encoding="utf-8-sig")

    if not args.no_plots:
        make_plots(tables, chi_df, perm_df, dim_df, args.output_dir)

    print(f"Loaded scopes: {', '.join(tables.keys())}")
    print(f"CSV outputs: {args.output_dir}")
    if not args.no_plots:
        print(f"Figures: {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
