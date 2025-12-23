# Enhanced Anomaly Detection in Keystroke Dynamics Authentication

This repository is part of the **CS266 - Topics in Information Security** course project titled **"Enhanced Anomaly Detection in Keystroke Dynamics Authentication"**. The project was submitted as part of the course requirements by the team members:
- Rashmi Sonth
- Sakshi Sanskruti Tripathy

The project focuses on analyzing different methodologies for anomaly detection in keystroke dynamics authentication, leveraging advanced machine learning and deep learning models.

---

## File Structure

### **1. Dataset**
- **`demographics.csv`**:
  - Contains demographic information of participants.
  - Used to supplement analysis with demographic-based features.
  
- **`free-text.csv`**:
  - Contains raw keystroke dynamics data.
  - Includes timing features such as `DU.key1.key1`, `DD.key1.key2`, etc.
  - This dataset is critical for building and testing machine learning models.

- **Dataset Source**:
  - The dataset used in this project can be downloaded from [Zenodo](https://zenodo.org/records/7886743).

### **2. Notebook**
- **`models.ipynb`**:
  - The main Jupyter Notebook for training and evaluating models.
  - Implements machine learning and deep learning approaches (e.g., LSTM, CNN).
  - **Important**: Ensure the dataset paths (`free-text.csv` and `demographics.csv`) are correctly updated in this notebook based on your local environment.

### **3. Output**
- **`output.png`**:
  - Helps to evaluate the effectiveness of implemented techniques.

---

## Instructions

1. **Update Dataset Paths**:
   - The datasets are located in the `dataset` folder.
   - Ensure you update the dataset paths in the `models.ipynb` file wherever necessary:
     ```python
     data = pd.read_csv('dataset/free-text.csv')
     demographics = pd.read_csv('dataset/demographics.csv')
     ```

2. **Run the Notebook**:
   - Open `models.ipynb` in Jupyter Notebook or any compatible environment.
   - Execute the cells step-by-step to train and evaluate the models.

---

## Requirements

- Python 3.7+
- Libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `sklearn`

  ---

  ## Quick Local Run

  1. Install dependencies:

  ```bash
  python -m pip install -r requirements.txt
  ```

  2. Run the main script (uses `dataset/free-text.csv` and `dataset/demographics.csv`):

  ```bash
  python models.py
  ```

  Note: the script will automatically fall back to the local `dataset/` folder when not running in Google Colab.

  ---

  ## Full Setup & Run (Windows)

  ### Option A — Recommended (conda, fastest & avoids build issues)

  1. Create and activate an environment with conda:

  ```powershell
  conda create -n keystroke python=3.11 -y
  conda activate keystroke
  ```

  2. Install binary packages from conda-forge:

  ```powershell
  conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib seaborn -y
  ```

  3. Run the script:

  ```powershell
  python models.py
  ```

  ### Option B — Virtualenv (works but may require build toolchain)

  1. Create the venv (PowerShell):

  ```powershell
  python -m venv .venv
  ```

  2. Use the venv's Python directly (no activation required):

  ```powershell
  .venv\bin\python.exe -m pip install --upgrade pip setuptools wheel
  .venv\bin\python.exe -m pip install -r requirements.txt
  .venv\bin\python.exe models.py
  ```

  Note: on Windows some venvs use `Scripts` instead of `bin` (check `.venv` for either `bin` or `Scripts`).

  ## Troubleshooting

  - **SSL / certificate verify failures while pip installing**: This environment shows `ssl.SSLCertVerificationError` when pip attempts to download build backends (cmake, ninja). Common fixes:
    - Use the Conda approach above (conda provides prebuilt binaries). This is recommended on Windows.
    - Ensure system time is correct and corporate proxies are configured. If your network intercepts TLS, install the organization's CA into the OS certificate store.
    - As a last resort, set the environment variable `PIP_DISABLE_PIP_VERSION_CHECK=1` and use `--trusted-host` for pip (not recommended for long-term use).

  - **Build errors for numpy/pandas**: Building `numpy` and `pandas` from source on Windows often requires a C/C++ toolchain (MSVC) and CMake. Use conda to avoid this.

  ## What I changed in this repo

  - **models.py**: added an automatic fallback to read CSVs from the local `dataset/` folder when Google Colab is not present. See [models.py](models.py).
  - **requirements.txt**: pinned a minimal set of packages for the project. See [requirements.txt](requirements.txt).
  - **This README**: expanded setup and troubleshooting steps. See [Readme.md](Readme.md).

  ## Next steps I can do for you

  - Convert the Jupyter notebook `models.ipynb` into a small runnable script with CLI flags to run only preprocessing or only evaluation.
  - Add a lightweight test that runs a smoke-check on the dataset load (no ML heavy installs).
  - Prepare a `environment.yml` for conda so others can reproduce the exact environment.

  If you want me to proceed with any of the above, tell me which one and I'll implement it.
