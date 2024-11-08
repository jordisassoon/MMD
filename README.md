# Statistical Experiments with Maximum Mean Discrepancy (MMD)

This project uses MMD (Maximum Mean Discrepancy) to conduct statistical experiments, such as detecting distributional differences in datasets. It includes tests based on Newcomb's speed of light measurements and explores the effects of kernel bandwidth, sample size, and outliers.

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Command-Line Options](#command-line-options)
  - [Example Commands](#example-commands)
- [Modules](#modules)

## Requirements

This project requires Python 3.7+ and the following libraries:

- `torch`
- `numpy`
- `scipy`
- `tqdm`
- `argparse`
- `pprint`

To install the required packages, run:

```bash
pip install torch numpy scipy tqdm argparse pprint matplotlib
```

## Setup

1. **Clone the repository** (if applicable) or download the project files.
2. **Data file**: Ensure the file `newcomb.txt` is available in the `data/` directory. This file should contain Newcomb's speed of light measurements.
3. **Run the main script** using the instructions below.

## Usage

The main script allows you to run different experiments using command-line arguments. To execute the main program, use:

```bash
python main.py --test <test_type> --ignore_outliers <True/False> --kernel <kernel_type>
```

### Command-Line Options

The main script accepts the following arguments:

- `--test <test_type>`: Specifies the type of test to run. Available options:
  - `newcomb`: Conducts an MMD test on Newcomb's speed of light data.
  - `sigma`: Tests how different values of `sigma` (kernel bandwidth) affect p-values.
  - `sample_size`: Examines the effect of varying sample sizes on p-values.
  - `outliers`: Evaluates the impact of outliers on the p-values.
  
- `--ignore_outliers <True/False>`: Indicates whether to ignore outliers in the data (default is `True`).
  
- `--kernel <kernel_type>`: Specifies the kernel to use for MMD calculations. Options:
  - `rbf`: Radial Basis Function (RBF) kernel
  - `lap`: Laplacian kernel (default)
  - `exp`: Exponential kernel

### Example Commands

Here are some examples of how to run specific tests:

1. **Newcomb's Speed of Light Test**:
   ```bash
   python main.py --test newcomb --ignore_outliers True --kernel rbf
   ```
   Runs the Newcomb test with outliers ignored and uses the RBF kernel.

2. **Varying Sigma Test**:
   ```bash
   python main.py --test sigma --kernel lap
   ```
   Explores how different sigma values impact p-values using the Laplacian kernel.

3. **Varying Sample Size Test**:
   ```bash
   python main.py --test sample_size --kernel lap
   ```
   Runs the sample size experiment with the Exponential kernel.

4. **Outliers Test**:
   ```bash
   python main.py --test outliers --kernel lap
   ```
   Investigates how the presence of outliers influences p-values using the Laplacian kernel.

## Modules

- **main.py**: The entry point of the project, which executes the specified tests.
- **plotter.py**: Contains functions to generate and display plots for visualizing test results.
- **kernels.py**: Implements kernel functions (`rbf_kernel`, `laplacian_kernel`, and `exponential_kernel`) for use in MMD calculations.

## Explanation of Core Functions

1. **set_seed**: Sets a random seed for reproducibility.
2. **load_newcomb**: Loads Newcomb's speed of light dataset, optionally removing outliers.
3. **mmd**: Calculates the MMD statistic between two distributions.
4. **compute_p**: Calculates the p-value from the MMD statistic through bootstrap sampling.
5. **witness_function**: Computes the witness function, visualizing differences between data distributions.
