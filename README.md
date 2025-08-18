# Shell Hackathon - Neuralnetics Team Submission

## Overview

The world is in dire need of innovative solutions to solve growing energy demands at a sustainable rate. This project is our submission for the Shell 2025 Hackathon, which challenged participants to predict 10 Blend Properties obtained from mixing 5 fuel components using machine learning algorithms.

## Public Leaderboard Score (MAPE) on 50% of testset: 95.47592%

---

## Project Structure

The repository is organized into two main directories: `final_submission`, which contains the complete competition workflow, and `prototype`, which holds the user-facing inference application. A detailed breakdown of the file structure is as follows:

```
.
├── LICENSE
└── Neuralnetics/
    ├── final_submission/
    │   ├── autofeat/                 # Custom module for automated feature engineering
    │   ├── dataset/                  # Raw competition datasets
    │   ├── lgb_shap/                 # Experimental artifacts from LightGBM models
    │   ├── plots/                    # Generated plots and visualizations
    │   ├── raw_FE/                   # Experimental artifacts from raw engineered features
    │   ├── X_test/                   # Preprocessed test sets for each model
    │   ├── X_train/                  # Preprocessed training sets for each model
    │   ├── shell_ai_ml_pipeline.ipynb  # <- START HERE: Main competition notebook
    │   └── submission.csv            # <- Final predictions for the public leaderboard
    └── prototype/
        ├── app.py                    # <- Main script to run the web app (streamlit run app.py)
        ├── config.py                 # App configurations and constants
        ├── core_logic.py             # <- Backend prediction logic for the app
        ├── ui_tabs.py                # Renders the UI components for the Streamlit app
        ├── features/                 # JSON files with feature lists for each model
        ├── models_full/              # Trained and serialized models for the app
        ├── pickles_for_inference.ipynb # <- Notebook to train and save models for the app
        ├── prototype.mp4             # <- Demo video of the web app in action
        └── requirements.txt          # <- Dependencies to run the prototype
```

---

## Setup and Installation

To ensure reproducibility, it is highly recommended to set up a dedicated `conda` environment with the exact package versions used in development.

1.  **Clone the Repository**: First, clone this repository to your local machine.

    ```bash
    git clone https://github.com/LD-Shell/shell_hackathon_2025.git
    cd shell_hackathon_2025
    ```

2.  **Create a Conda Environment**: This command creates a new environment named `shell-hackathon` with the specific versions of Python used and other core packages.

    ```bash
    conda create -n shell-hackathon python=3.11.13 pandas=2.2.3 numpy=2.1.3 scikit-learn=1.6.1 jupyterlab=4.4.4
    ```

3.  **Activate the Environment**:

    ```bash
    conda activate shell-hackathon
    ```

4.  **Install Core Libraries**: Install the main libraries for machine learning and data visualization with the specific versions used.

    ```bash
    conda install -c conda-forge torch=2.6.0 torchvision=0.21.0 lightgbm=4.6.0 xgboost=3.0.2 catboost=1.2.8 tabpfn=2.1.0 numexpr=2.11.0 matplotlib=3.10.3 seaborn=0.13.2 plotly=6.2.0
    ```

5.  **Install Prototype Dependencies**: Navigate to the `prototype` directory and install the required packages for the Streamlit application. The `requirements.txt` file contains the pinned versions needed for the app.

    ```bash
    cd Neuralnetics/prototype
    pip install -r requirements.txt
    ```

6.  **(Optional) GPU Support with RAPIDS**: If you have a CUDA-enabled GPU, you can install the specific versions of RAPIDS libraries used for significant acceleration.

    ```bash
    conda install -c rapidsai -c conda-forge -c nvidia cudf=25.06 cuml=25.06 cupy=13.5 dask-cuda=25.06
    ```

---

## Methodology

Our approach is broken down into four main stages: feature engineering, model training, submission preparation, and conclusion.

### 1. Feature Engineering Pipeline

* **Feature Engineering with Autofeat**: We used a modified instance of the AutoFeat library to generate complex feature combinations (order 2). Our version was adapted to leverage GPU acceleration for significantly faster computation. The original library can be found at <https://github.com/cod3licious/autofeat>. The model was fit on the training data and then used to transform both training and test sets.

* **Variance-Based Feature Selection**: We calculated the per-column variance to identify and remove the lowest-variance features, establishing a variance threshold to create a pruned feature set.

* **Model-Based Feature Importance**: To identify the most influential predictors for each target property, we computed gain-based feature importances using LightGBM (`importance_type="gain"`). This process was performed for each target (`BlendProperty1` → `BlendProperty10`) using the pruned feature set from the previous step.

* **Supervised Feature Selection using Cutoff Sweep**: For each target, we determined the minimum set of high-importance features that achieved the best validation performance. We iterated through different cutoff thresholds, training a `TabPFNRegressor` (developers here: <https://github.com/PriorLabs/TabPFN>) for each and evaluating the Mean Absolute Percentage Error (MAPE) on a validation split. This produced an optimized, target-specific feature set that balanced performance and dimensionality.

### 2. Model Training and Prediction

We developed three distinct solution pipelines to test different feature strategies.

* **Solution Pipeline 1: TabPFN with Hyperparameter Sweep (Using Selected Features)**: We trained a series of `TabPFNRegressor` models using the optimized feature subsets identified in the feature importance pipeline, combined with a grid search over key hyperparameters.

* **Solution Pipeline 2: TabPFN with Hyperparameter Sweep (Using Variance-Based Features Only)**: In this pipeline, we trained a separate `TabPFNRegressor` model for each of the 10 targets, using only the features that passed our initial variance-based selection process.

* **Solution Pipeline 3: TabPFN with Hyperparameter Sweep (Using Engineered Features Only)**: We performed an exhaustive hyperparameter sweep for the `TabPFNRegressor`, training a model for each target using the complete set of engineered features generated exclusively from the Autofeat process. This produced multiple submission candidates, saved under the `raw_FE/` directory.

### 3. Prepare Competitive Submission File from Predictions

For the final submission, we blended predictions from multiple strong pipelines to maximize performance.

We began with our highest-scoring TabPFN-based model as a base file. Then, using property-level performance analysis, we selectively replaced weaker predictions with those from other pipelines that demonstrated superior results for specific targets. The final blended file combined the most accurate property predictions from both TabPFN and LGBM-based pipelines. This targeted ensembling strategy provided a performance boost over any single pipeline.

### 4. Conclusion

We approached the challenge by combining aggressive feature engineering, targeted model training, and precision ensembling. Using Autofeat to generate complex interactions, we applied variance-based pruning and LightGBM gain-based feature selection to isolate the most informative predictors per property. We trained multiple `TabPFNRegressor` pipelines with different feature subsets and hyperparameters, then strategically blended the strongest predictions on a per-target basis, incorporating LightGBM outputs where they excelled. This targeted, property-level ensembling delivered a public leaderboard score of **95.47592**, surpassing the performance of any single pipeline.

---

## Inference Prototype

To make our models accessible, we developed an interactive web application using Streamlit. This prototype provides a user-friendly interface for uploading component data and receiving blend property predictions without requiring any knowledge of the underlying code.

### 1. Architecture and Design

The application is built with a modular architecture to separate concerns and improve maintainability:

* **`app.py`**: The main entry point for the application. It configures the Streamlit page, initializes the app's memory (`st.session_state`) to hold data across user interactions, and renders the main tab-based navigation.

* **`ui_tabs.py`**: Manages the user interface. It contains dedicated functions to render the content of each tab, separating the visual layout from the backend logic. This includes handling file uploads, configuration toggles, and results displays.

* **`core_logic.py`**: The backend engine of the application. This module contains the complete prediction pipeline, which includes:
    * Validating the input CSV to ensure all required columns are present.
    * Generating all necessary features on-the-fly from the base components.
    * Loading the serialized `.pkl` models.
    * Executing predictions for each selected target property.
    * Returning predictions or any errors encountered during the process.

* **`config.py`**: A centralized file for application constants, such as default file paths and UI text, allowing for easy configuration changes.

### 2. Workflow and Key Features

The prototype guides the user through a simple, three-step workflow powered by several key technologies:

* **Step 1: Setup & Upload**: The user configures the environment, choosing between CPU and GPU (if a CUDA-enabled device is available), and uploads their data as a CSV file.

* **Step 2: Configure & Predict**: The user selects which blend properties to predict and can choose to run inference on all rows or a specific subset.

* **Step 3: Results**: The application displays a summary of the run and a table of the resulting predictions, which can be downloaded as a new CSV file.

This seamless experience is enabled by:

* **Interactive UI with Streamlit**: The entire interface, including widgets, progress bars, and live logging, is built with Streamlit, which allows for the rapid development of interactive data applications.

* **State Management**: The application uses Streamlit's `session_state` to maintain data (like the uploaded DataFrame and prediction results) as the user navigates between tabs.

* **High-Performance Feature Engineering**: To ensure fast predictions, the feature generation process is optimized with the `NumExpr` library, which evaluates mathematical expressions at C-level speeds, significantly outperforming standard Pandas operations.

* **Device-Agnostic Model Loading**: The prototype uses a custom context manager (`torch_map_location`) to load PyTorch-based models seamlessly on either a CPU or a GPU, regardless of the device they were originally trained on. This prevents common device-mismatch errors and enhances portability.

---

## Contributors

* Daramola Olanrewaju
* Emmanuel Olanrewaju
* Israel Trejo

---

## License

This project is licensed under the terms specified in the `LICENSE` file.

