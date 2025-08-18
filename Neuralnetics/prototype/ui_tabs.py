# ui_tabs.py
# Author: Daramola
# Contains functions that render each of the main tabs in the Streamlit UI.

import time
import pandas as pd
import streamlit as st
from pathlib import Path

# Import our own modules
import config
import core_logic

def render_setup_tab():
    """Renders the 'Setup & Upload' tab."""
    st.header("Step 1: Configure Your Environment")
    p1, p2 = st.columns(2)
    
    with p1:
        st.subheader("Compute Settings")
        use_gpu = st.toggle("Use GPU (CUDA)", value=True, help="If available, run predictions on your NVIDIA GPU.")
        
        # Get device info and store it in the session state for other tabs to use.
        device, device_info = core_logic.get_device_info(use_gpu)
        st.session_state.device = device
        st.session_state.device_info = device_info
        
        if device == 'cuda':
            st.success(f"{config.ICONS['gpu']} GPU Mode Active: **{device_info}**")
        else:
            st.info(f"{config.ICONS['cpu']} CPU Mode Active: **{device_info}**")
            
    with p2:
        st.subheader("File Paths")
        # Store paths in session state as well.
        st.session_state.models_dir = Path(st.text_input("Models Directory", value=str(config.MODELS_DIR)))
        st.session_state.features_dir = Path(st.text_input("Features Directory", value=str(config.FEATURES_DIR)))

    st.divider()
    st.header("Step 2: Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file with component data.", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_upload = df # This is key for sharing data between tabs.
            st.success(f"Successfully loaded `{uploaded_file.name}` with **{len(df):,} rows** and **{len(df.columns)} columns**.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read the CSV file. Error: {e}")
            st.session_state.df_upload = None
    elif st.session_state.df_upload is not None:
        st.info("Using previously uploaded data. To change it, upload a new file.")

def render_predict_tab():
    """Renders the 'Configure & Predict' tab."""
    # Check if a file is uploaded; if not, stop the app here.
    if st.session_state.df_upload is None:
        st.warning(f"{config.ICONS['warning']} Please upload a CSV file in the 'Setup & Upload' tab first!")
        st.stop()

    df = st.session_state.df_upload
    st.header("Configure Your Prediction Task")

    # Use columns to lay out the row selection options.
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Row Selection")
        row_mode = st.radio("Predict On", ["All Rows", "Row Range"], horizontal=True, key="row_mode")
    
    # --- FIX 1: Ensure df_predict is always correctly set based on row_mode ---
    # This fixes the issue where df_predict would hold a stale slice
    # when switching back from "Row Range" to "All Rows".
    if row_mode == "Row Range":
        with c2:
            st.subheader("Range Definition")
            r1, r2 = st.columns(2)
            min_v, max_v = 0, len(df) - 1
            # Add unique keys to avoid widget conflicts.
            start_row = r1.number_input("Start Row (0-indexed)", min_value=min_v, max_value=max_v, value=min_v, key="start_row")
            end_row = r2.number_input("End Row (inclusive)", min_value=start_row, max_value=max_v, value=max_v, key="end_row")
            
        # Slice the DataFrame and assign it to df_predict.
        df_predict = df.iloc[start_row : end_row + 1]
    else:
        # If "All Rows" is selected, use the full DataFrame.
        df_predict = df

    st.info(f"**{len(df_predict):,}** rows selected for prediction.")
    st.divider()

    st.subheader("Target Selection")
    c1, c2 = st.columns(2)
    with c1:
        # Find the most likely ID column automatically.
        id_candidates = [c for c in df.columns if any(k in c.lower() for k in ['id', 'sample', 'identifier'])]
        default_id_idx = list(df.columns).index(id_candidates[0]) if id_candidates else 0
        id_col = st.selectbox("Sample ID Column", options=df.columns, index=default_id_idx, help="This column uniquely identifies each row and will be kept in the output.")
    with c2:
        available_targets = core_logic.load_target_properties(st.session_state.models_dir)
        if not available_targets:
            st.error(f"No models found in `{st.session_state.models_dir}`. Make sure `model_BlendProperty*.pkl` files are there.")
            st.stop()
        selected_targets = st.multiselect("Properties to Predict", options=available_targets, default=available_targets)
    
    st.divider()
    
    if st.button(f"{config.ICONS['blend']} Launch Prediction!", type="primary", use_container_width=True, disabled=(not selected_targets or df_predict.empty)):
        # Setup the UI elements for feedback.
        progress_bar = st.progress(0, text="Starting prediction...")
        status_text = st.empty()
        log_expander = st.expander("Live Prediction Log", expanded=True)
        log_container = log_expander.container()

        t_start = time.perf_counter()
        
        # Reset the index to prevent indexing mismatches
        original_df_index = df_predict.index # Save the original index before dropping columns.
        X_input = df_predict.drop(columns=[id_col], errors='ignore').reset_index(drop=True)

        # Call the main pipeline
        preds, failed, pred_idx = core_logic.predict_pipeline(
            X_df=X_input, 
            selected_targets=selected_targets, 
            models_dir=st.session_state.models_dir, 
            features_dir=st.session_state.features_dir, 
            device=st.session_state.device,
            ui_feedback=(progress_bar, status_text, log_container),
            column_names_path=config.COLUMN_NAMES_PATH
        )
        
        t_total = time.perf_counter() - t_start
        
        # Store results in session state for the results tab.
        st.session_state.last_run_info = {"device": st.session_state.device, "duration": t_total, "rows": len(df_predict)}
        st.session_state.failed_targets = failed
        
        if preds:
            # Create a DataFrame from predictions, using the original index for correct mapping.
            pred_df = pd.DataFrame(preds, index=original_df_index)
            # Get the ID column from the original, unsliced DataFrame using the saved index.
            id_series = df.loc[original_df_index, id_col]
            # Concatenate the ID and prediction dataframes, ensuring proper alignment.
            result_df = pd.concat([id_series, pred_df], axis=1)
            st.session_state.predictions = result_df
        else:
            st.session_state.predictions = None # Clear old predictions if the run fails completely

        # A final status message to guide the user.
        if preds:
             st.success(f"🎉 Success! Your predictions are ready. Navigate to the **{config.ICONS['results']} Results** tab to view and download.")
        else:
             st.warning(f"⚠️ Processing finished, but no predictions were successful. Check the log above and the **{config.ICONS['results']} Results** tab for error details.")

def render_results_tab():
    """Renders the 'Results' tab."""
    if not st.session_state.get('last_run_info'):
        st.info(f"{config.ICONS['info']} Your results will show up here once you run a prediction.")
        st.stop()

    # --- Display Run Summary ---
    info = st.session_state.last_run_info
    preds = st.session_state.get('predictions')
    failed = st.session_state.get('failed_targets', [])
    
    num_success = len(preds.columns) - 1 if preds is not None else 0
    num_failed = len(failed)
    
    st.header(f"Run Summary (Device: {info['device'].upper()})")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Time", core_logic.human_s(info['duration']))
    metric_cols[1].metric("Rows Processed", f"{info['rows']:,}")
    metric_cols[2].metric("Targets", f"{num_success} Succeeded, {num_failed} Failed")

    if preds is not None and not preds.empty:
        st.subheader("Prediction Results")
        st.dataframe(preds, use_container_width=True, height=400)
        
        # Download button for the results.
        csv_bytes = preds.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", data=csv_bytes,
                           file_name=f"predictions_{st.session_state.app_run_id}.csv",
                           mime="text/csv", use_container_width=True)
    else:
        st.warning(f"{config.ICONS['warning']} No predictions were successful for this run.")

    # --- Display Logs for Failed Targets ---
    if failed:
        with st.expander(f"{config.ICONS['error']} View {num_failed} Failed Target Logs", expanded=True):
            st.error("Some targets failed. Check errors below for clues (e.g., missing columns, library version mismatches).")
            st.dataframe(pd.DataFrame(failed), use_container_width=True)