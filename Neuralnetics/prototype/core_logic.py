# core_logic.py
# Author: Daramola
# Handles the core data processing and model prediction for the Blend Property Predictor app.

import os
import re
import json
import traceback
from pathlib import Path
import contextlib

import numpy as np
import pandas as pd
import joblib
import numexpr as ne
import streamlit as st
import torch

# A little helper to make time readouts more human-friendly.
def human_s(sec: float) -> str:
    """Formats time in seconds to a human-readable string."""
    return f"{sec:.2f}s" if sec < 120 else f"{sec/60:.2f} min"

# Use Streamlit's caching to avoid re-scanning the models directory on every interaction.
@st.cache_data(show_spinner="Scanning for available models...")
def load_target_properties(models_dir: Path):
    """Finds all 'BlendProperty' models in the specified directory."""
    if not models_dir.is_dir():
        return []
    # Find all model files and clean up the names for the UI dropdown.
    return sorted([p.stem.replace("model_", "") for p in models_dir.glob("model_BlendProperty*.pkl")])

@contextlib.contextmanager
def torch_map_location(device: str):
    """
    A context manager to safely load PyTorch models.
    This temporarily intercepts `torch.load` to inject a `map_location` argument, which
    is critical for loading a model saved on a GPU onto a CPU-only machine.
    """
    original_load = torch.load
    def _mapped_load(f, *args, **kwargs):
        # Force the model's tensors to be loaded onto the correct device.
        kwargs['map_location'] = torch.device(device)
        return original_load(f, *args, **kwargs)
    try:
        torch.load = _mapped_load
        yield
    finally:
        torch.load = original_load # Always restore the original function.

def get_device_info(use_gpu: bool) -> tuple:
    """Checks for a CUDA-enabled GPU and sets necessary environment variables."""
    device, device_name = ("cpu", "CPU")
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        # Ensure no GPUs are visible to PyTorch if we're in CPU mode.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if use_gpu:
             device_name = "CUDA requested but unavailable"
    # Set a specific flag for the TabPFN model to allow it to run on large datasets on the CPU.
    if device == 'cpu':
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"
    return device, device_name

def _predict_one_target_flexible(target: str, feats: list, model_path: str, X_all: pd.DataFrame, device: str):
    """
    Loads a single model and runs prediction on the input data.
    This function contains a two-stage fix to handle device incompatibilities.
    """
    try:
        # STAGE 1: Survive the Load.
        # The context manager ensures joblib.load doesn't crash when loading a GPU-trained
        # model on a CPU, by mapping all tensors to the target device.
        with torch_map_location(device):
            estimator = joblib.load(model_path)

        # STAGE 2: Correct the Model's Internal State.
        # If running on CPU, we explicitly patch the loaded model object to ensure
        # it knows it should operate in a CPU-only mode.
        if device == 'cpu':
            try:
                estimator.device = "cpu"
                if hasattr(estimator, "_model"):
                    # Deleting the cached model forces it to rebuild cleanly on the CPU.
                    estimator._model = None
            except Exception as patch_error:
                # This patch is specific to our TabPFN models; log if it fails.
                print(f"Info: Could not apply CPU patch for {target}: {patch_error}")

        # Ensure the model is on the correct device and in evaluation mode.
        if hasattr(estimator, 'to'):
            estimator.to(device)
        if hasattr(estimator, 'eval'):
            estimator.eval()

        # Step 3: Run prediction with torch.no_grad() for better performance.
        with torch.no_grad():
            needed_feats = getattr(estimator, "features", feats)
            X_model = X_all.reindex(columns=needed_feats, fill_value=0.0).to_numpy()
            y_pred = estimator.predict(X_model)

        return ("OK", target, y_pred.astype(np.float32))

    except Exception as e:
        # If anything goes wrong, catch it and return the error for display in the UI.
        error_details = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return ("ERR", target, error_details)

def build_union_features(targets: list, features_dir: Path) -> tuple:
    """Reads all required feature lists and creates a single, unique set."""
    union_features, target_features_map, missing_files = set(), {}, []
    for tgt in targets:
        prop_num = tgt.replace("BlendProperty", "")
        feature_file = features_dir / f"features_prop{prop_num}.json"
        if not feature_file.exists():
            missing_files.append(f"Feature definition file not found for {tgt}")
            continue
        try:
            with open(feature_file, "r") as f:
                feats = json.load(f).get("features", [])
            target_features_map[tgt] = feats
            # A set is used to automatically handle duplicates.
            union_features.update(feats)
        except json.JSONDecodeError:
            missing_files.append(f"Could not parse JSON for {tgt}")
    return target_features_map, sorted(list(union_features)), missing_files

def generate_features_fast(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Efficiently generates a feature matrix from a list of feature expressions.
    Uses NumExpr for fast, C-level evaluation of mathematical operations.
    """
    # Create a dictionary of numpy arrays for fast lookups by NumExpr.
    env = {col: df[col].to_numpy(dtype=np.float32) for col in df.columns}
    
    # To avoid slow, repetitive DataFrame appends (which causes fragmentation),
    # we compute all feature arrays first and store them in a dictionary.
    new_cols = {}
    for feature_expr in feature_list:
        if feature_expr in df.columns:
            # If it's a base feature, just copy it over.
            new_cols[feature_expr] = df[feature_expr].to_numpy(dtype=np.float32)
        else:
            # Otherwise, evaluate the expression.
            try:
                arr = ne.evaluate(feature_expr, local_dict=env)
                new_cols[feature_expr] = arr
            except Exception:
                # Fallback to pandas.eval for safety, though it's slightly slower.
                try:
                    series = df.eval(feature_expr, engine="numexpr")
                    new_cols[feature_expr] = series.to_numpy(dtype=np.float32)
                except Exception:
                    # If a feature can't be generated, fill it with NaNs.
                    nan_arr = np.full(df.shape[0], np.nan, dtype=np.float32)
                    new_cols[feature_expr] = nan_arr
    
    # Create the final DataFrame in a single, efficient operation.
    generated_features = pd.DataFrame(new_cols, index=df.index)
    
    # Reindex to ensure correct column order and .copy() for a defragmented memory layout.
    return generated_features.reindex(columns=feature_list).copy()

def validate_input_columns(df: pd.DataFrame, column_names_path: Path) -> list:
    """Checks if the user's uploaded CSV contains all the necessary base columns."""
    try:
        with open(column_names_path, "r") as f:
            required_columns = json.load(f)
        # Use set difference for a fast check of missing columns.
        return list(set(required_columns) - set(df.columns))
    except FileNotFoundError:
        return [f"Error: column_names.json not found at {column_names_path}"]
    except json.JSONDecodeError:
        return [f"Error: Could not parse column_names.json."]

def predict_pipeline(X_df, selected_targets, models_dir, features_dir, device, ui_feedback, column_names_path: Path):
    """
    The main prediction workflow that orchestrates all steps from data validation
    to returning final predictions.
    """
    progress_bar, status_text, log_container = ui_feedback
    failed_tasks = []

    # Step 1: Validate the input data to fail fast if columns are missing.
    status_text.text("🧐 Validating input file columns...")
    missing_cols = validate_input_columns(X_df, column_names_path)
    if missing_cols:
        error_msg = f"❌ The input file is missing required columns: {', '.join(missing_cols)}."
        status_text.error(error_msg)
        with log_container: st.error(error_msg)
        progress_bar.progress(1.0)
        return {}, failed_tasks, X_df.index
    
    # Step 2: Prepare the master feature matrix.
    status_text.text("⚙️ Preparing data and features...")
    X_numeric = X_df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    target_to_feats, union_feats, feature_errors = build_union_features(selected_targets, features_dir)
    failed_tasks.extend([{"target": "Feature Loading", "error": err} for err in feature_errors])
    with log_container:
        st.write("Live Log:")
        for f in failed_tasks: st.error(f"🚨 {f['target']}: {f['error']}")
            
    X_all = generate_features_fast(X_numeric, union_feats)
    # Basic imputation for any remaining missing values.
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    impute_values = X_all.median(numeric_only=True).fillna(0)
    X_all = X_all.fillna(impute_values).astype(np.float32)
    
    # Step 3: Prepare the list of prediction tasks.
    task_args = []
    for tgt in selected_targets:
        if tgt in target_to_feats:
            model_path = models_dir / f"model_{tgt}.pkl"
            if model_path.exists():
                task_args.append((tgt, target_to_feats[tgt], str(model_path), X_all))
                
    task_args.sort(key=lambda arg: int(re.search(r'(\d+)$', arg[0]).group(1) or 0))
    if not task_args:
        status_text.warning("No valid models could be loaded for the selected targets.")
        progress_bar.progress(1.0)
        return {}, failed_tasks, X_all.index

    # Step 4: Execute predictions sequentially for stability.
    # The parallel implementation was removed to guarantee correctness and avoid
    # complex bugs related to object serialization and state management.
    predictions = {}
    total_tasks = len(task_args)
    status_text.text(f"Processing tasks sequentially on {device.upper()}...")

    for i, (tgt, feats, path, X_all_for_task) in enumerate(task_args):
        progress_text = f"Predicting {tgt} on {device.upper()}... ({i + 1}/{total_tasks})"
        status_text.text(progress_text)
        progress_bar.progress((i + 1) / total_tasks, text=progress_text)
        
        status, target, data = _predict_one_target_flexible(tgt, feats, path, X_all_for_task, device)
        
        if status == "OK":
            predictions[target] = data
            log_container.success(f"✅ Success: {target}")
        else:
            failed_tasks.append({"target": target, "error": data})
            log_container.error(f"🚨 Failed: {target} | Error: {data}")
            
    status_text.success("✅ All prediction tasks complete!")
    return predictions, failed_tasks, X_all.index