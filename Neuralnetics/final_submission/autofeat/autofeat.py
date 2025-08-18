# autofeat.py (Fully Accelerated with LightGBM)

# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT
# GPU Orchestrator & Vectorization by Gemini & Dexter
# LightGBM Integration & Correlation Filter by Gemini & Dexter

from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd
import pint
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sympy.utilities.lambdify import lambdify

# Assuming feateng and featsel are in the same directory or installed
from feateng import colnames2symbols, engineer_features, n_cols_generated
from featsel import select_features

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

# --- Helper for Optional Imports ---
def _check_and_print_import_status(module_name, purpose, install_command):
    """
    Checks for a module, prints a helpful message on failure, and returns the module or None.
    """
    try:
        module = __import__(module_name)
        logging.info(f"[AutoFeat] Successfully imported {module_name} for {purpose}.")
        return module, True
    except ImportError:
        logging.warning(f"[AutoFeat] Could not import `{module_name}`.")
        logging.warning(f"  - Purpose: {purpose}.")
        logging.warning(f"  - Impact: The library will fall back to slower, CPU-based operations or functionality will be limited.")
        logging.warning(f"  - To install, run: `{install_command}`")
        return None, False

# --- GPU/Optimized Library Imports ---
# Check for each optional dependency and provide clear feedback.
cp, CUPY_AVAILABLE = _check_and_print_import_status("cupy", "GPU-accelerated array operations (like NumPy)", "pip install cupy-cuda11x")
cudf, CUDF_AVAILABLE = _check_and_print_import_status("cudf", "GPU-accelerated DataFrames (like pandas)", "conda install -c rapidsai -c conda-forge -c nvidia cudf")
numexpr, NUMEXPR_AVAILABLE = _check_and_print_import_status("numexpr", "Optimized CPU expression evaluation", "pip install numexpr")
lightgbm, LIGHTGBM_AVAILABLE = _check_and_print_import_status("lightgbm", "High-performance gradient boosting models (CPU/GPU)", "pip install lightgbm")


def _parse_units(units: dict, ureg: pint.UnitRegistry | None = None, verbose: int = 0):
    parsed_units = {}
    if units:
        if ureg is None:
            ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
        for c in units:
            try:
                parsed_units[c] = ureg.parse_expression(units[c])
            except pint.UndefinedUnitError:
                if verbose > 0:
                    logging.warning(f"[AutoFeat] unit {units[c]} of column {c} was not recognized and will be ignored!")
                parsed_units[c] = ureg.parse_expression("")
            parsed_units[c].__dict__["_magnitude"] = 1.0
    return parsed_units

class AutoFeatModel(BaseEstimator):
    def __init__(
        self,
        problem_type: str = "regression",
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        corr_threshold: float | None = 0.98,
        max_gb: int | None = None,
        lgbm_params: dict | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        use_gpu: bool = True,
        verbose: int = 0,
    ):
        self.problem_type = problem_type
        self.categorical_cols = categorical_cols
        self.feateng_cols = feateng_cols
        self.units = units
        self.feateng_steps = feateng_steps
        self.featsel_runs = featsel_runs
        self.corr_threshold = corr_threshold
        self.max_gb = max_gb
        self.lgbm_params = lgbm_params
        self.transformations = transformations
        self.apply_pi_theorem = apply_pi_theorem
        self.always_return_numpy = always_return_numpy
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.verbose = verbose

    def __getstate__(self):
        return {k: self.__dict__[k] if k != "feature_functions_" else {} for k in self.__dict__}

    def _transform_categorical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        self.categorical_cols_map_ = {}
        if self.categorical_cols:
            e = OneHotEncoder(sparse_output=False, categories="auto", handle_unknown='ignore')
            for c in self.categorical_cols:
                if c not in df.columns:
                    raise ValueError(f"[AutoFeat] categorical_col {c} not in df.columns")
                ohe = e.fit_transform(df[c].to_numpy()[:, None])
                new_cat_cols = [f"cat_{c}_{val}" for val in e.categories_[0]]
                self.categorical_cols_map_[c] = new_cat_cols
                df = df.join(pd.DataFrame(ohe, columns=new_cat_cols, index=df.index))
            df = df.drop(columns=self.categorical_cols)
        return df

    def _apply_pi_theorem(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.apply_pi_theorem and self.units:
            ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
            parsed_units = _parse_units(self.units, ureg, self.verbose)
            q = {c: parsed_units[c] for c in self.feateng_cols_ if c in parsed_units and not parsed_units[c].dimensionless}
            if self.verbose:
                logging.info("[AutoFeat] Applying the Pi Theorem")
            pi_theorem_results = ureg.pi_theorem(q)
            for i, r in enumerate(pi_theorem_results, 1):
                if self.verbose:
                    logging.info(f"[AutoFeat] Pi Theorem {i}: {pint.formatter(r.items(), single_denominator=True)}")
                cols = sorted(r)
                not_na_idx = df[cols].notna().all(axis=1)
                ptr = df.loc[not_na_idx, cols[0]].to_numpy() ** r[cols[0]]
                for c in cols[1:]:
                    ptr *= df.loc[not_na_idx, c].to_numpy() ** r[c]
                df.loc[not_na_idx, f"PT{i}_{pint.formatter(r.items(), single_denominator=True).replace(' ', '')}"] = ptr
        return df

    def _filter_correlated_features_gpu(self, df: pd.DataFrame) -> list:
        """
        Calculates the correlation matrix on the GPU and returns a list of columns to keep.
        """
        if not CUDF_AVAILABLE:
            if self.verbose:
                logging.warning("[AutoFeat] cuDF not available for GPU correlation check. Skipping.")
            return list(df.columns)

        if self.verbose:
            logging.info(f"[AutoFeat] Performing GPU-accelerated correlation filtering with threshold {self.corr_threshold}...")
        
        # Move data to GPU
        cudf_df = cudf.from_pandas(df)
        
        # Calculate correlation matrix on GPU
        corr_matrix = cudf_df.corr()
        
        # Move back to CPU for iteration (iteration is faster on CPU)
        corr_matrix_cpu = corr_matrix.to_pandas()
        
        # Create a set of columns to drop
        cols_to_drop = set()
        
        # Iterate over the correlation matrix to find highly correlated pairs
        for i in range(len(corr_matrix_cpu.columns)):
            for j in range(i):
                col_i = corr_matrix_cpu.columns[i]
                col_j = corr_matrix_cpu.columns[j]
                if col_i in cols_to_drop or col_j in cols_to_drop:
                    continue
                if abs(corr_matrix_cpu.iloc[i, j]) > self.corr_threshold:
                    # If correlated, decide to drop one. We'll drop the one that came later in the list.
                    cols_to_drop.add(col_i)

        # Determine the columns to keep
        cols_to_keep = [c for c in df.columns if c not in cols_to_drop]
        
        if self.verbose:
            logging.info(f"[AutoFeat] Dropped {len(cols_to_drop)} highly correlated features.")
            
        return cols_to_keep

    def _generate_features(self, df: pd.DataFrame, new_feat_cols: list) -> pd.DataFrame:
        check_is_fitted(self, ["feature_formulas_"])
        if not new_feat_cols:
            return df

        use_gpu_backend = self.use_gpu and CUDF_AVAILABLE
        if self.verbose:
            backend = "GPU (cuDF.eval)" if use_gpu_backend else "CPU (numexpr)"
            logging.info(f"[AutoFeat] Computing {len(new_feat_cols)} new features using vectorized backend: {backend}.")

        if use_gpu_backend:
            df_eval = cudf.from_pandas(df[self.feateng_cols_])
            # --- FIX for CUDF failure ---
            # Explicitly cast all columns to a consistent float type to prevent
            # type mismatch errors in cudf.eval(). float32 is usually sufficient.
            for col in df_eval.columns:
                df_eval[col] = df_eval[col].astype('float32')
        else:
            if not NUMEXPR_AVAILABLE:
                raise ImportError("`numexpr` is not installed. Please install it for optimized CPU feature generation.")
            df_eval = df[self.feateng_cols_]

        new_features_df = pd.DataFrame(index=df.index)
        for i, new_col_name in enumerate(new_feat_cols):
            if self.verbose:
                print(f"[AutoFeat] {i+1:5}/{len(new_feat_cols):5} new features", end="\r")
            symbolic_expr = str(self.feature_formulas_[new_col_name])
            try:
                if use_gpu_backend:
                    new_col_data = df_eval.eval(symbolic_expr)
                    new_features_df[new_col_name] = new_col_data.to_pandas()
                else:
                    local_dict = {col: df_eval[col].values for col in df_eval.columns}
                    new_features_df[new_col_name] = numexpr.evaluate(symbolic_expr, local_dict=local_dict)
            except Exception as e:
                if self.verbose:
                    logging.warning(f"\n[AutoFeat] Could not generate feature {new_col_name} with expression '{symbolic_expr}'. Error: {e}. Filling with NaNs.")
                new_features_df[new_col_name] = np.nan
        if self.verbose:
            print(f"[AutoFeat] {len(new_feat_cols):5}/{len(new_feat_cols):5} new features ...done.")
        return df.join(new_features_df)

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # --- FIX for FutureWarning ---
        # Changed `force_all_finite` to `ensure_all_finite=False`
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None, ensure_all_finite=False)
        if not cols:
            cols = [f"x{i:03}" for i in range(X.shape[1])]
        self.original_columns_ = cols
        df = pd.DataFrame(X, columns=cols)
        df = self._transform_categorical_cols(df)

        self.feateng_cols_ = [item for c in self.feateng_cols or list(df.columns) for item in self.categorical_cols_map_.get(c, [c])]

        if self.units:
            self.units = {c: self.units.get(c, "") for c in self.feateng_cols_}
            df = self._apply_pi_theorem(df)

        n_cols = n_cols_generated(len(self.feateng_cols_), self.feateng_steps, len(self.transformations))
        n_gb = (len(df) * n_cols) / 250_000_000
        if self.verbose: logging.info(f"[AutoFeat] The {self.feateng_steps}-step process could generate up to {n_cols} features, requiring ~{n_gb:.2f}GB memory.")

        if self.max_gb and n_gb > self.max_gb:
            n_rows = int(self.max_gb * 250_000_000 / n_cols)
            if self.verbose: logging.info(f"[AutoFeat] Subsampling to {n_rows} data points to stay below memory limit of {self.max_gb:.1f}GB.")
            subsample_idx = np.random.permutation(list(df.index))[:n_rows]
            df_subs = df.loc[subsample_idx].reset_index(drop=True)
            target_sub = target[subsample_idx]
        else:
            df_subs = df.copy()
            target_sub = target.copy()

        df_subs, self.feature_formulas_ = engineer_features(
            df_subs, self.feateng_cols_, _parse_units(self.units, verbose=self.verbose),
            self.feateng_steps, self.transformations, self.verbose, use_gpu=self.use_gpu,
        )

        # Pre-filter highly correlated features on GPU before main feature selection
        if self.use_gpu and self.corr_threshold is not None:
            good_cols_initial = self._filter_correlated_features_gpu(df_subs)
            df_subs = df_subs[good_cols_initial]

        if self.featsel_runs <= 0:
            if self.verbose: logging.warning("[AutoFeat] Not performing feature selection.")
            good_cols = list(df_subs.columns)
        else:
            good_cols = select_features(
                df_subs, target_sub, self.featsel_runs, None, self.problem_type, self.n_jobs, self.use_gpu, self.verbose,
            )
            if not good_cols: good_cols = list(df.columns)

        self.new_feat_cols_ = [c for c in good_cols if c not in list(df.columns)]
        self.good_cols_ = good_cols
        df = self._generate_features(df, self.new_feat_cols_)
        df.columns = [str(c) for c in df.columns]
        self.feature_formulas_.update({f: self.feature_formulas_[f] for f in self.new_feat_cols_})
        self.all_columns_ = list(df.columns)

        if self.verbose: logging.info(f"[AutoFeat] Final dataframe with {len(df.columns)} features ({len(self.new_feat_cols_)} new).")

        # --- Model Selection with LightGBM ---
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed, which is required for the final model. Please run `pip install lightgbm`.")

        params = self.lgbm_params or {}
        # Set default parameters and override with user-provided ones
        default_params = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 20, "n_jobs": self.n_jobs}
        default_params.update(params)
        
        if self.use_gpu:
            default_params["device"] = "gpu"

        if self.verbose: logging.info(f"[AutoFeat] Using LightGBM for final {self.problem_type} model.")
        
        if self.problem_type == "regression":
            model = lightgbm.LGBMRegressor(**default_params)
        elif self.problem_type == "classification":
            model = lightgbm.LGBMClassifier(**default_params)
        else:
            model = None

        if model is not None:
            if self.verbose: logging.info(f"[AutoFeat] Training final {self.problem_type} model.")
            X_final = df[self.good_cols_].fillna(0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_final, target)
            self.prediction_model_ = model
            if self.problem_type == "classification": self.classes_ = model.classes_

        if self.always_return_numpy: return df.to_numpy()
        return df

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        _ = self.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self, ["feature_formulas_"])
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else [f"x{i:03}" for i in range(X.shape[1])]
        if not cols == self.original_columns_:
            raise ValueError("[AutoFeat] Not the same features as when calling fit.")

        # --- FIX for FutureWarning ---
        df = pd.DataFrame(check_array(X, ensure_all_finite=False, dtype=None), columns=cols)
        df = self._transform_categorical_cols(df)
        df = self._apply_pi_theorem(df)
        df = self._generate_features(df, self.new_feat_cols_)
        df.columns = [str(c) for c in df.columns]

        if self.always_return_numpy:
            return df.to_numpy()
        return df

    def _X2df(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else [f"x{i:03}" for i in range(X.shape[1])]
        # --- FIX for FutureWarning ---
        df = pd.DataFrame(check_array(X, dtype=None, ensure_all_finite=False), columns=cols)
        if list(df.columns) != self.all_columns_:
            return self.transform(df)
        return df

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["prediction_model_"])
        df = self._X2df(X)
        X_pred_df = df[self.good_cols_].fillna(0)
        return self.prediction_model_.predict(X_pred_df)

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> float:
        check_is_fitted(self, ["prediction_model_"])
        df = self._X2df(X)
        X_score_df = df[self.good_cols_].fillna(0)
        return self.prediction_model_.score(X_score_df, y)


class AutoFeatRegressor(AutoFeatModel, RegressorMixin):
    def __init__(self, *, categorical_cols: list | None = None, feateng_cols: list | None = None, units: dict | None = None,
                 feateng_steps: int = 2, featsel_runs: int = 5, corr_threshold: float | None = 0.98, max_gb: int | None = None, lgbm_params: dict | None = None,
                 transformations: tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                 apply_pi_theorem: bool = True, always_return_numpy: bool = False, n_jobs: int = 1,
                 use_gpu: bool = True, verbose: int = 0):
        super().__init__(problem_type="regression", categorical_cols=categorical_cols, feateng_cols=feateng_cols,
                         units=units, feateng_steps=feateng_steps, featsel_runs=featsel_runs, corr_threshold=corr_threshold, max_gb=max_gb, lgbm_params=lgbm_params,
                         transformations=transformations, apply_pi_theorem=apply_pi_theorem,
                         always_return_numpy=always_return_numpy, n_jobs=n_jobs, use_gpu=use_gpu, verbose=verbose)


class AutoFeatClassifier(AutoFeatModel, ClassifierMixin):
    def __init__(self, *, categorical_cols: list | None = None, feateng_cols: list | None = None, units: dict | None = None,
                 feateng_steps: int = 2, featsel_runs: int = 5, corr_threshold: float | None = 0.98, max_gb: int | None = None, lgbm_params: dict | None = None,
                 transformations: tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                 apply_pi_theorem: bool = True, always_return_numpy: bool = False, n_jobs: int = 1,
                 use_gpu: bool = True, verbose: int = 0):
        super().__init__(problem_type="classification", categorical_cols=categorical_cols, feateng_cols=feateng_cols,
                         units=units, feateng_steps=feateng_steps, featsel_runs=featsel_runs, corr_threshold=corr_threshold, max_gb=max_gb, lgbm_params=lgbm_params,
                         transformations=transformations, apply_pi_theorem=apply_pi_theorem,
                         always_return_numpy=always_return_numpy, n_jobs=n_jobs, use_gpu=use_gpu, verbose=verbose)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["prediction_model_"])
        df = self._X2df(X)
        X_pred_df = df[self.good_cols_].fillna(0)
        return self.prediction_model_.predict_proba(X_pred_df)
