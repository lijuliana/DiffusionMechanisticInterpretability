"""
Variance decomposition for multivariate outcomes (e.g. token embeddings) by categorical factors.

Used by the T5 / shape notebooks to build ``effect_vecs`` and permutation p-values.
Implementation is faithful to the notebook contract (marginal / partial SS on squared Euclidean distance).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _design_matrix(factors: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    """Full-rank dummy design (drop first level per factor)."""
    names = list(factors.keys())
    dfX = pd.DataFrame({k: pd.Series(v).astype("category") for k, v in factors.items()})
    levels_map: dict[str, np.ndarray] = {k: dfX[k].cat.categories.values for k in names}
    dummies = pd.get_dummies(dfX, columns=names, drop_first=True, dtype=float)
    return dummies, levels_map, dummies.shape[1]


def _ss_resid(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares ||Y - X @ B||_F^2 for least squares."""
    if X.shape[1] == 0:
        y_mean = y.mean(axis=0, keepdims=True)
        diff = y - y_mean
        return float(np.sum(diff * diff))
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    diff = y - pred
    return float(np.sum(diff * diff))


def variance_partition_with_effects(
    Y: np.ndarray,
    factors: Mapping[str, Any],
    *,
    metric: str = "euclidean",
    n_perm: int = 100,
    verbose: bool = True,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """
    Parameters
    ----------
    Y : (n_samples, n_dim)
    factors : dict[str, Series|array] — categoricals aligned with rows of Y.

    Returns
    -------
    var_part_df, intercept, effect_vecs, levels_map, R2_total
    """
    if metric != "euclidean":
        raise ValueError("Only metric='euclidean' is supported.")

    Y = np.asarray(Y, dtype=np.float64)
    n, d = Y.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    rng = np.random.default_rng(random_state)
    factor_names = list(factors.keys())

    fullX, levels_map, _p_full = _design_matrix(factors)
    X_full = np.c_[np.ones(n), fullX.to_numpy(dtype=np.float64)]

    y_mean = Y.mean(axis=0)
    intercept = y_mean.copy()
    Yc = Y - y_mean
    ss_total = float(np.sum(Yc * Yc))

    ss_res_full = _ss_resid(Y, X_full)
    r2_total = 1.0 - ss_res_full / ss_total if ss_total > 0 else 0.0

    effect_vecs: dict[str, np.ndarray] = {}

    rows: list[dict[str, Any]] = []
    for fname in factor_names:
        cats = pd.Series(factors[fname]).astype("category")
        levels = cats.cat.categories.values
        codes = cats.cat.codes.to_numpy()
        grand = Y.mean(axis=0)
        gmeans = np.stack([Y[codes == i].mean(axis=0) if np.any(codes == i) else grand for i in range(len(levels))])
        effect_vecs[fname] = (gmeans - grand).astype(np.float64)

        # Marginal model: intercept + this factor only
        Xj = pd.get_dummies(cats, drop_first=True, dtype=float).to_numpy(dtype=np.float64)
        Xj = np.c_[np.ones(n), Xj]
        ss_res_j = _ss_resid(Y, Xj)
        ss_marginal = ss_total - ss_res_j
        r2_marginal = ss_marginal / ss_total if ss_total > 0 else 0.0

        # Partial (unique) SS: full minus reduced without this factor
        other_factors = {k: factors[k] for k in factor_names if k != fname}
        if other_factors:
            redX, _, _ = _design_matrix(other_factors)
            X_red = np.c_[np.ones(n), redX.to_numpy(dtype=np.float64)]
        else:
            X_red = np.ones((n, 1))
        ss_res_red = _ss_resid(Y, X_red)
        ss_partial = ss_res_red - ss_res_full
        if ss_partial < 0 and ss_partial > -1e-6:
            ss_partial = 0.0
        r2_partial = ss_partial / ss_total if ss_total > 0 else 0.0
        denom_partial = ss_partial + ss_res_full
        eta2_partial = ss_partial / denom_partial if denom_partial > 1e-12 else 0.0

        # Permutation p-value for partial SS
        observed = ss_partial
        if n_perm <= 0:
            p_perm = float("nan")
        else:
            hits = 0
            codes_perm = codes.copy()
            for _ in range(n_perm):
                rng.shuffle(codes_perm)
                fac_perm = pd.Series(pd.Categorical.from_codes(codes_perm, categories=levels))
                tmp_factors = dict(factors)
                tmp_factors[fname] = fac_perm
                # rebuild reduced / full with permuted column
                idx = factor_names.index(fname)
                names_red = [factor_names[i] for i in range(len(factor_names)) if i != idx]
                if names_red:
                    redX_p, _, _ = _design_matrix({k: tmp_factors[k] for k in names_red})
                    X_red_p = np.c_[np.ones(n), redX_p.to_numpy(dtype=np.float64)]
                else:
                    X_red_p = np.ones((n, 1))
                ss_rf_p = _ss_resid(Y, X_red_p)
                fullX_p, _, _ = _design_matrix(tmp_factors)
                X_f_p = np.c_[np.ones(n), fullX_p.to_numpy(dtype=np.float64)]
                ss_ff_p = _ss_resid(Y, X_f_p)
                ss_p = ss_rf_p - ss_ff_p
                if ss_p >= observed:
                    hits += 1
            p_perm = (hits + 1) / (n_perm + 1)

        df_eff = max(len(levels) - 1, 0)
        df_resid = max(n - X_full.shape[1], 0)
        rows.append(
            {
                "feature": fname,
                "levels": int(len(levels)),
                "df_effect": df_eff,
                "df_resid": df_resid,
                "SS_total": ss_total,
                "SSR_marginal": ss_marginal,
                "R2_marginal": r2_marginal,
                "SSR_partial": ss_partial,
                "R2_partial": r2_partial,
                "eta2_partial": eta2_partial,
                "p_partial_perm": p_perm,
            }
        )

    var_part_df = pd.DataFrame(rows)
    if verbose:
        print(f"Total R2 (all features): {r2_total:.6f}")
        print(f"Total R² (all features): {r2_total:.4f}")

    if np.any(var_part_df["SSR_partial"].values < -1e-4):
        warnings.warn("Negative partial SSR values; check collinear factors or sample size.")

    return var_part_df, intercept, effect_vecs, levels_map, r2_total
