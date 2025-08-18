# featsel.py - v2, Hardened by Dexter's Review

from __future__ import annotations

import logging
import warnings
from collections import Counter

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from nb_utils import nb_standard_scale

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def _standard_scale(X):
    """
    Scales the data using the appropriate backend (NumPy or CuPy).
    Uses cp.get_array_module for robust type detection.
    """
    xp = cp.get_array_module(X) if CUPY_AVAILABLE else np
    if xp is np:
        return nb_standard_scale(X)
    else:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        return (X - mean) / (std + 1e-8)


def _add_noise_features(X):
    """
    Adds noise features to X using memory-efficient per-column shuffling.
    Accepts either NumPy or CuPy array.
    """
    xp = cp.get_array_module(X) if CUPY_AVAILABLE else np
    n_feat = X.shape[1]

    if X.shape[0] > 50 and n_feat > 1:
        # Per-column permutation is much more memory-friendly than flatten()
        shuffled_cols = [xp.random.permutation(X[:, j]) for j in range(n_feat)]
        shuffled = xp.vstack(shuffled_cols).T
        rand_noise = _standard_scale(shuffled)
        X = xp.hstack([X, rand_noise])

    rand_noise_normal = xp.random.randn(X.shape[0], max(3, int(0.5 * n_feat)))
    return xp.hstack([X, rand_noise_normal])


def _noise_filtering(
    X: np.ndarray,
    target: np.ndarray,
    good_cols: list | None = None,
    problem_type: str = "regression",
    use_gpu: bool = False,
    seed: int = 42,
) -> list:
    """
    Noise filtering that now accepts a seed for reproducible noise generation.
    """
    xp = cp if use_gpu and CUPY_AVAILABLE else np
    np.random.seed(seed)
    if xp is cp:
        cp.random.seed(seed)

    n_feat = X.shape[1]
    if good_cols is None: good_cols = list(range(n_feat))
    assert len(good_cols) == n_feat

    if problem_type == "regression": model = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification": model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        logging.warning(f"[featsel] Unknown problem_type {problem_type} - not performing noise filtering.")
        return good_cols

    X_processed = xp.asarray(X)
    X_processed = _add_noise_features(X_processed)

    if xp is cp:
        X_processed = cp.asnumpy(X_processed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_processed, target)
        except ValueError:
            rand_idx = np.random.permutation(X_processed.shape[0])
            model.fit(X_processed[rand_idx], target[rand_idx])

    coefs = np.abs(model.coef_) if problem_type == "regression" else np.max(np.abs(model.coef_), axis=0)
    weights = dict(zip(good_cols, coefs[: len(good_cols)]))
    noise_w_thr = np.max(coefs[n_feat:])
    return [c for c in good_cols if weights[c] > noise_w_thr]


def _select_features_1run(
    df_scaled: pd.DataFrame,
    target: np.ndarray,
    problem_type: str = "regression",
    use_gpu: bool = False,
    verbose: int = 0,
    seed: int = 42,
) -> list:
    """
    A single feature selection run, now fully reproducible via a seed.
    """
    if df_scaled.shape[0] <= 1: raise ValueError(f"n_samples = {df_scaled.shape[0]}")

    xp = cp if use_gpu and CUPY_AVAILABLE else np
    np.random.seed(seed)
    if xp is cp:
        cp.random.seed(seed)

    if problem_type == "regression": model1 = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification": model1 = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        logging.warning(f"[featsel] Unknown problem_type {problem_type} - not performing feature selection!")
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model1.fit(df_scaled, target)
        except ValueError:
            rand_idx = np.random.permutation(df_scaled.shape[0])
            model1.fit(df_scaled.iloc[rand_idx], target[rand_idx])

    coefs1 = np.abs(model1.coef_) if problem_type == "regression" else np.max(np.abs(model1.coef_), axis=0)
    thr = sorted(coefs1, reverse=True)[min(df_scaled.shape[1] - 1, df_scaled.shape[0] // 5)]
    initial_cols = list(df_scaled.columns[coefs1 > thr])

    initial_cols = _noise_filtering(df_scaled[initial_cols].to_numpy(), target, initial_cols, problem_type, use_gpu, seed)
    good_cols_set = set(initial_cols)
    if verbose > 0: logging.info(f"[featsel]\t {len(initial_cols)} initial features.")
    
    if not initial_cols:
        if verbose > 0: logging.info("\n[featsel]\t No initial features found, stopping early.")
        return []

    X_initial_gpu = xp.asarray(df_scaled[initial_cols].to_numpy())
    X_w_noise_gpu = _add_noise_features(X_initial_gpu)

    other_cols = list(np.random.permutation(list(set(df_scaled.columns).difference(initial_cols))))
    if other_cols:
        n_splits = int(np.ceil(len(other_cols) / max(10, 0.5 * df_scaled.shape[0] - len(initial_cols))))
        split_size = int(np.ceil(len(other_cols) / n_splits))
        for i in range(n_splits):
            current_cols = other_cols[i * split_size : min(len(other_cols), (i + 1) * split_size)]
            X_current_gpu = xp.asarray(df_scaled[current_cols].to_numpy())
            X_iter_gpu = xp.hstack([X_current_gpu, X_w_noise_gpu])

            X_iter_cpu = cp.asnumpy(X_iter_gpu) if xp is cp else X_iter_gpu

            if problem_type == "regression": model2 = lm.LassoLarsCV(cv=5, eps=1e-8)
            else: model2 = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model2.fit(X_iter_cpu, target)
                except ValueError:
                    rand_idx = np.random.permutation(X_iter_cpu.shape[0])
                    model2.fit(X_iter_cpu[rand_idx], target[rand_idx])

            current_cols.extend(initial_cols)
            coefs2 = np.abs(model2.coef_) if problem_type == "regression" else np.max(np.abs(model2.coef_), axis=0)
            weights = dict(zip(current_cols, coefs2[: len(current_cols)]))
            noise_w_thr = np.max(coefs2[len(current_cols):])
            good_cols_set.update([c for c in weights if abs(weights[c]) > noise_w_thr])
            if verbose > 0: print(f"[featsel]\t Split {i + 1:2}/{n_splits}: {len(good_cols_set):3} candidate features identified.", end="\r")

    good_cols = list(good_cols_set)
    if not good_cols: return []
    good_cols = _noise_filtering(df_scaled[good_cols].to_numpy(), target, good_cols, problem_type, use_gpu, seed)
    if verbose > 0: logging.info(f"\n[featsel]\t Selected {len(good_cols):3} features after noise filtering.")
    return good_cols


def select_features(
    df: pd.DataFrame,
    target: np.ndarray,
    featsel_runs: int = 5,
    keep: list | None = None,
    problem_type: str = "regression",
    n_jobs: int = 1,
    use_gpu: bool = True,
    verbose: int = 0,
) -> list:
    if not (len(df) == len(target)): raise ValueError("[featsel] df and target dimension mismatch.")
    if keep is None: keep = []
    keep = [c for c in keep if c in df.columns and str(c) not in df.columns] + [str(c) for c in keep if str(c) in df.columns]

    xp = cp if use_gpu and CUPY_AVAILABLE else np

    if verbose > 0:
        if featsel_runs > df.shape[0]: logging.warning("[featsel] Less data points than featsel runs!!")
        print(f"[featsel] Scaling data using {xp.__name__}...", end="")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled_data = _standard_scale(xp.asarray(df.to_numpy()))
        if xp is cp:
            scaled_data = cp.asnumpy(scaled_data)
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns, dtype=np.float32)
        target_scaled = nb_standard_scale(target.reshape(-1, 1)).ravel() if problem_type == "regression" else target
    if verbose > 0: print("done.")

    def run_select_features(i: int):
        seed = i
        if verbose > 0: logging.info(f"[featsel] Feature selection run {i + 1}/{featsel_runs} (seed={seed})")
        np.random.seed(seed)
        rand_idx = np.random.permutation(df_scaled.index)[: max(10, int(0.85 * len(df_scaled)))]
        return _select_features_1run(df_scaled.iloc[rand_idx], target_scaled[rand_idx], problem_type, use_gpu, verbose - 1, seed)

    if featsel_runs < 1 or problem_type not in ("regression", "classification"):
        return list(df.columns)

    if n_jobs == 1 or featsel_runs == 1:
        selected_columns = [col for i in range(featsel_runs) for col in run_select_features(i)]
    else:
        def flatten_lists(l: list): return [item for sublist in l for item in sublist]
        selected_columns = flatten_lists(Parallel(n_jobs=n_jobs, verbose=100 * verbose)(delayed(run_select_features)(i) for i in range(featsel_runs)))

    if selected_columns:
        selected_columns_counter = Counter(selected_columns)
        selected_columns = sorted(selected_columns_counter, key=lambda x: selected_columns_counter[x] - 1e-6 * len(str(x)), reverse=True)
        if verbose > 0: logging.info(f"[featsel] {len(selected_columns)} features after {featsel_runs} feature selection runs")

        selected_columns = keep + [c for c in selected_columns if c not in keep]
        good_cols = keep[:]
        if len(selected_columns) > len(keep):
            if not keep:
                good_cols.append(selected_columns[0])
            
            corr_data = xp.asarray(df_scaled[selected_columns].to_numpy())
            corr_matrix = xp.corrcoef(corr_data, rowvar=False)
            if xp is cp:
                corr_matrix = cp.asnumpy(corr_matrix)
            
            for i in range(len(keep), len(selected_columns)):
                # correlation against columns already in good_cols
                col_indices = [selected_columns.index(c) for c in good_cols]
                if np.max(np.abs(corr_matrix[i, col_indices])) < 0.9:
                    good_cols.append(selected_columns[i])
        if verbose > 0: logging.info(f"[featsel] {len(good_cols)} features after correlation filtering")
    else:
        good_cols = keep[:]

    if good_cols:
        good_cols = _noise_filtering(df_scaled[good_cols].to_numpy(), target_scaled, good_cols, problem_type, use_gpu, seed=featsel_runs)
    if verbose > 0:
        logging.info(f"[featsel] {len(good_cols)} features after noise filtering")
        if not good_cols: logging.warning("[featsel] Not a single good feature was found...")

    good_cols = keep + [c for c in good_cols if c not in keep]
    if verbose > 0 and keep: logging.info(f"[featsel] {len(good_cols)} final features selected (including {len(keep)} original keep features).")
    return good_cols


class FeatureSelector(BaseEstimator):
    def __init__(self, problem_type: str = "regression", featsel_runs: int = 5, keep: list | None = None, n_jobs: int = 1, use_gpu: bool = True, verbose: int = 0):
        self.problem_type = problem_type
        self.featsel_runs = featsel_runs
        self.keep = keep
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.verbose = verbose

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        self.return_df_ = isinstance(X, pd.DataFrame)
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X.shape[1])]
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression")
        self.original_columns_ = cols
        df = pd.DataFrame(X, columns=cols)
        
        self.good_cols_ = select_features(df, target, self.featsel_runs, self.keep, self.problem_type, self.n_jobs, self.use_gpu, self.verbose)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self, ["good_cols_"])
        if not self.good_cols_:
            if self.verbose > 0: logging.warning("[FeatureSelector] No good features found; returning data unchanged.")
            return X
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X.shape[1])]
        X = check_array(X, force_all_finite="allow-nan")
        if not cols == self.original_columns_: raise ValueError("[FeatureSelector] Not the same features as when calling fit.")
        new_X = pd.DataFrame(X, columns=cols)[self.good_cols_]
        return new_X.to_numpy() if not self.return_df_ else new_X

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)