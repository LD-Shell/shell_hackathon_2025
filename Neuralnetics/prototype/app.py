# app.py
# Author: Daramola
# The main entry point for the Streamlit application.
# To run: streamlit run app.py

import uuid
import streamlit as st

# Import our separated modules
import config
import ui_tabs

# --- 1. App Configuration & Initialization ---
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session_state (memory of the app across user interactions)
if 'app_run_id' not in st.session_state:
    st.session_state.app_run_id = str(uuid.uuid4())[:8]
    st.session_state.df_upload = None
    st.session_state.predictions = None
    st.session_state.failed_targets = None
    st.session_state.last_run_info = {}
    # We also initialize keys that will be set in other tabs
    st.session_state.device = 'cpu'
    st.session_state.device_info = ''
    st.session_state.models_dir = config.MODELS_DIR
    st.session_state.features_dir = config.FEATURES_DIR

# --- 2. Main App Layout ---
st.title(f"{config.APP_TITLE} {config.ICONS['blend']}")
st.caption(f"{config.TEAM_INFO} (Session ID: {st.session_state.app_run_id})")

# Create the tabs for our multi-page interface.
tab1, tab2, tab3 = st.tabs([
    f"**1. Setup & Upload** {config.ICONS['setup']}", 
    f"**2. Configure & Predict** {config.ICONS['predict']}", 
    f"**3. Results** {config.ICONS['results']}"
])

# --- 3. Render Each Tab ---
# Each 'with' block corresponds to a tab, and we just call the
# appropriate rendering function from our ui_tabs module.
with tab1:
    ui_tabs.render_setup_tab()

with tab2:
    ui_tabs.render_predict_tab()

with tab3:
    ui_tabs.render_results_tab()