# Physics-Informed Neural Control (PINC) for Learning Drone Dynamics

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=PyTorch&logoColor=white)
![Tests Passed](https://img.shields.io/badge/tests-passed-brightgreen)



This repository implements a Physics-Informed Neural Network with Control (PINC) approach to model and simulate the dynamics of a drone using PyTorch. It leverages the power of neural networks while incorporating physical principles to achieve data-efficient neural dynamic models.


> **📄 Accepted Paper @ [IJCNN 2025](https://2025.ijcnn.org/)**  
> **Title:** *Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control*


## Key Features

*   **Drone Dynamic Simulation**: Core simulation logic for the drone implemented in `src/drone_model.py` and `src/drone_models/`.
*   **Synthetic Data Generation**: Scripts (`data/create_drone_data.py`) generate customizable trajectory datasets for training and evaluation.
*   **PINC Model Implementation**: Neural network models designed to learn system dynamics, incorporating physical laws directly into the loss function (PINN) or network structure. See [Model Explanation](model_explanation.md) for details on the architecture and loss functions.
*   **Model Training Pipeline**: Robust training script (`training/train_model.py`) using PyTorch, with support for hyperparameter tuning.


## Project Structure

```
.
├── data/             # Scripts for data generation and utility functions
├── memory-bank/      # Project documentation (context, progress, etc.)
├── models/           # Saved model checkpoints and utility functions. Contains trained neural network models.
├── notebooks/        # Jupyter notebooks for experimentation (potentially outdated)
├── results/          # Saved evaluation results, plots, summaries
├── runs/             # TensorBoard log files
├── scripts/          # Evaluation, analysis, and visualization scripts
├── src/              # Core source code (BlueROV model, parameters)
├── training/         # Model training scripts
├── .gitignore        # Git ignore rules
├── LICENSE           # Project license file (Should be updated to GPLv3)
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Installation

### Prerequisites

*   Python 3.8+
*   Git
*   PyTorch (>=2.0 recommended)
*   Optional: CUDA-compatible GPU for accelerated training

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/abdelhakim96/underwater_pinns.git
    cd underwater_pinns
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
    *Core Dependencies:*
    *   `torch`: Neural network framework
    *   `numpy`: Numerical computing
    *   `scipy`: Scientific computing tools
    *   `pandas`: Data manipulation and analysis
    *   `matplotlib`: Plotting and visualization
    *   `tqdm`: Progress bars
    *   `control`: Control system analysis library

## Usage (BlueROV Model)

*Note: The following instructions apply to the original BlueROV model.*

### 1. Generate BlueROV Data

Run the BlueROV data generation script. You might need to adjust parameters within the script.

```bash
python data/create_data_2.py
```

### 2. Train the BlueROV Model

Execute the training script, specifying the BlueROV model and data if necessary (though defaults might work).

```bash
# Example (check script defaults or modify as needed)
python training/train_model.py --model_name bluerov --data_dir training_set
```

### 3. Monitor BlueROV Training with TensorBoard (Optional)

TensorBoard provides powerful visualizations to monitor the training process in real-time or analyze it afterward. This helps in understanding model convergence, comparing different runs, and debugging potential issues.

To launch TensorBoard, run the following command in your terminal from the project's root directory:

```bash
tensorboard --logdir runs
```

This command points TensorBoard to the `runs/` directory where the training script saves its log files.

Once TensorBoard is running, open your web browser and navigate to the URL provided in the terminal output (usually `http://localhost:6006/`). You should see visualizations including:

*   Loss curves (training and validation)
*   Accuracy or other relevant metrics over epochs
*   Potentially model graphs and hyperparameter comparisons (depending on logging implementation in `training/train_model.py`)

### 4. Evaluate and Analyze Models

Use the scripts provided to evaluate performance and visualize results:

*   **Evaluate Metrics:**
    ```bash
    python scripts/evaluate_model.py --model_path path/to/your/model.pt
    ```
*   **Analyze Rollouts:**
    ```bash
    python scripts/analyze_rollout.py --model_path path/to/your/model.pt
    ```
*   **Visualize Predictions (Example):**
    ```bash
    python scripts/pi_dnn.py # (May need adjustments to load specific models/data)
    ```
*(Note: You might need to modify these scripts to point to the specific BlueROV model checkpoints you want to analyze, typically found in `models/`)*

## Usage (Drone Model)

*Note: The following instructions apply to the drone model workflow.*

### 1. Generate Drone Data

Run the drone data generation script. This uses a PID controller to generate trajectories (circle, lemniscate, sinusoidal hover).

```bash
python data/create_drone_data.py
```
This will create `training_set.pt` and `dev_set.pt` in the `drone_data/` directory.

### 2. Train the Drone Model

Execute the training script, ensuring it uses the drone model configuration and the correct data directory.

```bash
# Uses defaults: --model_name drone --data_dir drone_data
python training/train_model.py
```
Model checkpoints (including the best based on dev loss) will be saved in the `models/` directory (e.g., `drone_training_1_best_dev_l_XXX`).

### 3. Monitor Drone Training with TensorBoard (Optional)

Similar to the BlueROV model, use TensorBoard to monitor training progress:

```bash
tensorboard --logdir runs
```
Look for runs named like `drone_training_...`.

### 4. Evaluate the Drone Model

The `scripts/pi_dnn.py` script is currently configured to evaluate the drone model. It performs single-step and rollout predictions and generates plots.

```bash
python scripts/pi_dnn.py
```
*(Note: You might need to modify `scripts/pi_dnn.py` to load a specific trained drone model checkpoint if it doesn't automatically load the latest or best one. Results are saved in `results/drone_pinn_results/`)*

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
