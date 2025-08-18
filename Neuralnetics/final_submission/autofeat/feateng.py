# feateng.py

# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT
# GPU Accelerated and Patched by Gemini & Dexter

from __future__ import annotations
import logging
import operator as op
import re
from functools import reduce
from itertools import combinations, product
from typing import Callable

# Optional GPU library imports
CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    pass

CUML_AVAILABLE = False
if CUPY_AVAILABLE:
    try:
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        CUML_AVAILABLE = True
    except ImportError:
        pass
        
import numpy as np
import pandas as pd
import pint
import sympy
from sklearn.preprocessing import StandardScaler
from sympy.utilities.lambdify import lambdify

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

def colnames2symbols(c: str | int, i: int = 0) -> str:
    c = str(c)
    c = re.sub(r"\W+", "", c)
    if not c:
        c = f"x{i:03}"
    elif c[0].isdigit():
        c = "x" + c
    return c

def ncr(n: int, r: int) -> int:
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom

def n_cols_generated(n_features: int, max_steps: int, n_transformations: int = 8, n_combinations: int = 4) -> int:
    n_transformations -= 1
    original_cols = n_features
    new_cols, new_new_cols, n_additions, steps = 0, 0, 0, 1
    if steps <= max_steps:
        original_cols += n_features * n_transformations
        steps += 1
    if steps <= max_steps:
        new_cols = n_combinations * (ncr(original_cols, 2))
        n_additions += 3 * new_cols // 4
        steps += 1
    while steps <= max_steps:
        new_cols += new_cols * n_transformations
        steps += 1
        if steps <= max_steps:
            new_new_cols = n_combinations * (original_cols * new_cols)
            n_additions += 3 * new_new_cols // 4
            steps += 1
        if steps <= max_steps:
            n = n_combinations * (ncr(new_cols, 2))
            new_new_cols += n
            n_additions += 3 * n // 4
            steps += 1
            original_cols += new_cols
            new_cols = new_new_cols
            new_new_cols = 0
    if steps <= max_steps:
        new_cols += new_cols * n_transformations
    return original_cols + new_cols + new_new_cols - n_additions

def engineer_features(
    df_org: pd.DataFrame,
    start_features: list | None = None,
    units: dict | None = None,
    max_steps: int = 3,
    transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
    verbose: int = 0,
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, dict]:
    
    if use_gpu and CUPY_AVAILABLE:
        xp = cp
        if verbose > 0: logging.info("[feateng] 🚀 Using GPU (CuPy) for acceleration!")
    else:
        xp = np
        if verbose > 0 and use_gpu: logging.warning("[feateng] CuPy not found. Falling back to CPU (NumPy).")

    if start_features is None:
        start_features = list(df_org.columns)
    else:
        for c in start_features:
            if c not in df_org.columns:
                raise ValueError(f"[feateng] start feature {c} not in df_org.columns")
    
    feature_pool = {c: sympy.symbols(colnames2symbols(c, i), real=True) for i, c in enumerate(start_features)}
    
    if max_steps < 1:
        if verbose > 0: logging.warning("[feateng] no features generated for max_steps < 1.")
        return df_org, feature_pool
        
    df = pd.DataFrame(df_org.copy(), dtype=np.float32)

    def compile_func_transform(name: str, ft: Callable):
        t = sympy.symbols("t")
        expr_temp = ft(t)
        return lambdify(t, expr_temp, modules=[xp])

    def apply_transformations(features_list: list) -> tuple[list, set]:
        # Add nonlocal to fix the UnboundLocalError
        nonlocal df, feature_pool, units
        
        func_transform = {
            "exp": sympy.exp, "exp-": lambda x: sympy.exp(-x), "log": sympy.log,
            "abs": sympy.Abs, "sqrt": sympy.sqrt, "sin": sympy.sin, "cos": sympy.cos,
            "2^": lambda x: 2**x, "^2": lambda x: x**2, "^3": lambda x: x**3,
            "1+": lambda x: 1 + x, "1-": lambda x: 1 - x, "1/": lambda x: 1 / x,
        }
        func_transform_units = {
            "exp": np.exp, "exp-": np.exp, "log": np.log, "abs": np.abs, "sqrt": np.sqrt,
            "sin": np.sin, "cos": np.cos, "2^": np.exp, "^2": lambda x: x**2,
            "^3": lambda x: x**3, "1+": lambda x: 1 + x, "1-": lambda x: 1 - x,
            "1/": lambda x: 1 / x,
        }
        func_transform_cond = {
            "exp": lambda x: xp.all(x < 10), "exp-": lambda x: xp.all(-x < 10),
            "log": lambda x: xp.all(x > 1e-10), "abs": lambda x: xp.any(x < 0),
            "sqrt": lambda x: xp.all(x >= 0), "sin": lambda x: True, "cos": lambda x: True,
            "2^": lambda x: xp.all(x < 50),
            "^2": lambda x: xp.all(xp.abs(x) < 1_000_000),
            "^3": lambda x: xp.all(xp.abs(x) < 10_000),
            "1+": lambda x: True, "1-": lambda x: True, "1/": lambda x: xp.all(x != 0),
        }

        compiled_func_transformations = {k: compile_func_transform(k, v) for k, v in func_transform.items()}
        new_features, uncorr_features = [], set()
        feat_array = xp.zeros((df.shape[0], len(features_list) * len(transformations)), dtype=np.float32)
        cat_features = {feat for feat in features_list if len(df[feat].unique()) <= 2}
        
        for i, feat in enumerate(features_list):
            if verbose and not i % 100: print(f"[feateng] {i+1:15}/{len(features_list):15} features transformed", end="\r")
            if feat in cat_features: continue
            
            feat_data = xp.asarray(df[feat].to_numpy())
            for ft in transformations:
                if ft in func_transform and func_transform_cond[ft](feat_data):
                    expr = func_transform[ft](feature_pool[feat])
                    expr_name = str(expr)
                    if expr_name not in feature_pool:
                        if units:
                            try:
                                units[expr_name] = func_transform_units[ft](units[feat])
                                units[expr_name].__dict__["_magnitude"] = 1.0
                            except (pint.DimensionalityError, pint.OffsetUnitCalculusError):
                                continue
                        feature_pool[expr_name] = expr
                        f = compiled_func_transformations[ft]
                        new_feat = f(feat_data).astype(xp.float32)
                        
                        if xp.all(xp.isfinite(new_feat)) and xp.var(new_feat) > 1e-10:
                            corr = abs(xp.corrcoef(new_feat, feat_data)[0, 1])
                            if corr < 1.0:
                                feat_array[:, len(new_features)] = new_feat
                                new_features.append(expr_name)
                                if corr < 0.95:
                                    uncorr_features.add(expr_name)
        if verbose > 0:
            logging.info(f"[feateng] Generated {len(new_features)} transformed features... done.")
        
        if len(new_features) > 0:
            final_feats = cp.asnumpy(feat_array[:, :len(new_features)]) if xp is cp else feat_array[:, :len(new_features)]
            df = df.join(pd.DataFrame(final_feats, columns=new_features, index=df.index, dtype=np.float32))
        return new_features, uncorr_features

    def get_feature_combinations(feature_tuples: list) -> tuple[list, set]:
        # Add nonlocal to fix the UnboundLocalError
        nonlocal df, feature_pool, units
        
        func_combinations = {"x+y": lambda x, y: x + y, "x*y": lambda x, y: x * y, "x-y": lambda x, y: x - y}
        compiled_func_combinations = {k: lambdify((sympy.symbols("s"), sympy.symbols("t")), v(sympy.symbols("s"), sympy.symbols("t")), modules=[xp]) for k,v in func_combinations.items()}

        combinations_to_run = ["x*y"] if steps == max_steps else list(func_combinations.keys())
        new_features, uncorr_features = [], set()
        feat_array = xp.zeros((df.shape[0], len(feature_tuples) * len(combinations_to_run)), dtype=np.float32)
        gpu_cache = {}

        for i, (feat1, feat2) in enumerate(feature_tuples):
            if verbose and not i % 100: print(f"[feateng] {i+1:15}/{len(feature_tuples):15} feature tuples combined", end="\r")

            if feat1 not in gpu_cache: gpu_cache[feat1] = xp.asarray(df[feat1].to_numpy())
            if feat2 not in gpu_cache: gpu_cache[feat2] = xp.asarray(df[feat2].to_numpy())
            
            for fc in combinations_to_run:
                expr = func_combinations[fc](feature_pool[feat1], feature_pool[feat2])
                expr_name = str(expr)
                if expr_name not in feature_pool:
                    new_feat = compiled_func_combinations[fc](gpu_cache[feat1], gpu_cache[feat2]).astype(xp.float32)
                    if xp.all(xp.isfinite(new_feat)) and xp.var(new_feat) > 1e-10:
                        corr = max(abs(xp.corrcoef(new_feat, gpu_cache[feat1])[0,1]), abs(xp.corrcoef(new_feat, gpu_cache[feat2])[0,1]))
                        if corr < 1.0:
                            feature_pool[expr_name] = expr
                            feat_array[:, len(new_features)] = new_feat
                            new_features.append(expr_name)
                            if corr < 0.95: uncorr_features.add(expr_name)
        if verbose > 0:
            logging.info(f"[feateng] Generated {len(new_features)} feature combinations... done.")
        if len(new_features) > 0:
            final_feats = cp.asnumpy(feat_array[:, :len(new_features)]) if xp is cp else feat_array[:, :len(new_features)]
            df = df.join(pd.DataFrame(final_feats, columns=new_features, index=df.index, dtype=np.float32))
        return new_features, uncorr_features

    # Main feature engineering loop
    steps = 1
    original_features = list(feature_pool.keys())
    uncorr_features = set(feature_pool.keys())
    temp_new, temp_uncorr = apply_transformations(original_features)
    original_features.extend(temp_new)
    uncorr_features.update(temp_uncorr)
    steps += 1
    if steps <= max_steps:
        new_features, temp_uncorr = get_feature_combinations(list(combinations(original_features, 2)))
        uncorr_features.update(temp_uncorr)
        steps += 1
    while steps <= max_steps:
        temp_new, temp_uncorr = apply_transformations(new_features)
        new_features.extend(temp_new)
        uncorr_features.update(temp_uncorr)
        steps += 1
        if steps <= max_steps:
            new_new_features, temp_uncorr = get_feature_combinations(list(product(original_features, new_features)))
            uncorr_features.update(temp_uncorr)
            steps += 1
        if steps <= max_steps:
            temp_new, temp_uncorr = get_feature_combinations(list(combinations(new_features, 2)))
            new_new_features.extend(temp_new)
            uncorr_features.update(temp_uncorr)
            steps += 1
            original_features.extend(new_features)
            new_features = new_new_features

    feature_pool = {c: feature_pool[c] for c in feature_pool if c in uncorr_features and feature_pool[c].func != sympy.core.add.Add}
    cols = [c for c in df.columns if c in feature_pool and c not in df_org.columns]
    
    if cols:
        if use_gpu and CUML_AVAILABLE:
            if verbose > 0: logging.info("[feateng] Using GPU (cuML) for final correlation check.")
            new_cols_gpu = xp.asarray(df[cols].to_numpy(dtype=np.float32))
            org_gpu = xp.asarray(df_org[start_features].to_numpy(dtype=np.float32))
            new_cols_gpu[cp.isnan(new_cols_gpu)] = 0
            org_gpu[cp.isnan(org_gpu)] = 0
            scaled_cols_gpu = cuStandardScaler().fit_transform(new_cols_gpu)
            scaled_org_gpu = cuStandardScaler().fit_transform(org_gpu)
            corrs_gpu = cp.max(cp.abs(cp.dot(scaled_cols_gpu.T, scaled_org_gpu) / org_gpu.shape[0]), axis=1)
            corrs = dict(zip(cols, cp.asnumpy(corrs_gpu)))
        else:
            if verbose > 0 and use_gpu: logging.warning("[feateng] cuML not found. Final correlation check will be CPU-bound.")
            df_cols_np = df[cols].fillna(0).to_numpy()
            df_org_np = df_org[start_features].fillna(0).to_numpy()
            scaled_cols = StandardScaler().fit_transform(df_cols_np)
            scaled_org = StandardScaler().fit_transform(df_org_np)
            corrs_np = np.max(np.abs(np.dot(scaled_cols.T, scaled_org) / df_org.shape[0]), axis=1)
            corrs = dict(zip(cols, corrs_np))
        
        cols = [c for c in cols if corrs.get(c, 0) < 0.9]
    
    final_cols = start_features + cols
    
    if verbose > 0: logging.info(f"[feateng] Selected {len(final_cols) - len(start_features)} final features after correlation filtering.")
        
    return df[final_cols], feature_pool