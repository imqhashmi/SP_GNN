# SP_GNN

This repository contains the implementation of a Graph Neural Network (GNN) based approach for synthetic population generation.

Three stages — **individuals**, **households**, and **person-to-household assignment** — are trained per MSOA against published census tables. Preprocessed census CSVs cover England and Wales, so the same models can be run for:

- **Oxford** — 17 MSOA pilot areas (`E02005940`–`E02005957`, excluding `E02005952`)
- **Greater London** — all **983** MSOA areas (already present in `data/preprocessed-data/`)

<!-- 
## Requirements

- Python 3.10 or higher
- Dependencies can be installed via pip:
  ```
  pip install -r requirements.txt
  ``` -->

## Project Structure

```
SP_GNN/
├── code/
│   ├── generateHouseholds.py       # Household generation logic
│   ├── generateIndividuals.py      # Individual generation logic
│   ├── assignHouseholds.py         # Household assignment logic
│   ├── main.py                     # Interactive menu (Oxford + Greater London)
│   ├── runPipeline.py              # End-to-end batch runner for many areas
│   ├── evaluation.py               # Model evaluation utilities
│   ├── Utils/                      # Utility scripts directory
│   │   ├── greaterLondonAreas.py   # 983 London MSOA codes + 5-area subset
│   │   ├── createGlossary.py       # Create master glossary file
│   │   ├── plotConvergencePerformance.py  # Generate plots and analysis
│   │   ├── runAssignmentHPTuning.py       # Batch hyperparameter tuning
│   │   └── runMultipleAreas.py            # Batch processing utility (Oxford)
│   └── outputs/                    # Generated outputs directory
├── data/
│   ├── raw-data/                   # Original input data
│   ├── preprocessed-data/          # Processed and prepared data
│   └── encode_data.py              # Data encoding utilities
├── GreaterLondonSyntheticPopulation/  # Optional collected tensors from runPipeline
└── requirements.txt                # Project dependencies
```

## Directory Description

- `code/`: Contains the main implementation files for the GNN-based synthetic population generation
  - `generateHouseholds.py`: Implements the household generation algorithms
  - `generateIndividuals.py`: Implements the individual population generation
  - `assignHouseholds.py`: Contains logic for household assignment and hyperparameter optimization
  - `main.py`: Interactive menu for running all components; area selection supports Oxford and Greater London
  - `runPipeline.py`: Runs individuals → households → assignment across many areas (London by default), with optional multi-GPU workers and resume
  - `evaluation.py`: Model evaluation and comparison utilities
  - `Utils/`: Utility scripts for batch processing, analysis, and visualization
    - `greaterLondonAreas.py`: Canonical list of 983 Greater London MSOA codes and the five-area representative subset
    - `createGlossary.py`: Creates master glossary files from crosstables
    - `plotConvergencePerformance.py`: Generates convergence and performance plots
    - `runAssignmentHPTuning.py`: Batch hyperparameter tuning for all Oxford areas
    - `runMultipleAreas.py`: Batch processing for multiple Oxford geographical areas
  - `outputs/`: Stores the generated outputs and results (`individuals_<area>/`, `households_<area>/`, `assignment_hp_tuning_<area>/`)

- `data/`: Houses all data-related files and scripts
  - `raw-data/`: Contains the original, unprocessed input data
  - `preprocessed-data/`: Stores the processed and prepared data for model input (includes Oxford and Greater London rows)
  - `encode_data.py`: Utilities for data encoding and preprocessing

## Getting Started

1 - Ensure you have Python 3.10 or higher installed

2 - Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
3 - Installing Pytorch

GPU:

To install pytorch with cuda support; find you cuda version and install pytorch for that version i.e. for cuda 11.8:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
CPU: 

If you want to use pytorch with CPU then run below command:
   ```
   pip install torch==2.7.0
   ```
4 - Raw and Preprocessed data is in the `data/` directory

5 - Run the main menu interface to access all functionality:
   ```
   cd code
   python main.py
   ```
   
   The main menu provides access to:
   - Individual, household generation and household assignments for specific areas
   - Batch processing for multiple areas
   - Convergence and performance analysis
   - Model evaluation and comparison utilities

## Area selection (Oxford and Greater London)

Interactive runs that need an area code (`main.py` options 1–4) first ask for a **region**, then an MSOA:

1. **Oxford** — numbered list of the 17 pilot codes (you can also type any recognised MSOA code)
2. **Greater London** — shortlist of five representative areas (smallest → largest by population), or type any London MSOA code in full (e.g. `E02000001`)

The five-area shortlist and the full London list live in `code/Utils/greaterLondonAreas.py`:

| Code | Note |
|------|------|
| `E02000800` | smallest |
| `E02000691` | lower quartile |
| `E02000245` | median |
| `E02000282` | upper quartile |
| `E02000123` | largest |

Non-interactive example:

```bash
cd code
python main.py --script 1 --area_code E02000001
```

`--area_code` accepts any Oxford or Greater London MSOA present in the preprocessed tables. No extra data copy is required for London — those rows are already in `data/preprocessed-data/`.

## End-to-end pipeline (`runPipeline.py`)

Use this to run the full three-stage training pipeline over many areas (default: all 983 Greater London MSOAs). Stages run in order **per area**; areas can run in parallel across workers/GPUs.

```bash
cd code

# all 983 Greater London areas (GPUs detected automatically)
python runPipeline.py --region london

# five-area representative subset
python runPipeline.py --region subset

# Oxford pilot (17 areas)
python runPipeline.py --region oxford

# custom list / resume / dry-run
python runPipeline.py --areas E02000800,E02000123
python runPipeline.py --region london --skip-existing
python runPipeline.py --region subset --dry-run --no-collect
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--region london\|oxford\|subset` | Built-in area list (default: `london`) |
| `--areas` / `--areas-file` | Explicit codes (overrides `--region`) |
| `--stages` | Subset of `individuals,households,assignment` |
| `--workers` / `--gpus` / `--per-gpu` | Parallelism and device selection |
| `--skip-existing` | Resume: skip stages whose sentinel outputs already exist |
| `--collect-dir` / `--no-collect` | Copy primary tensors into `GreaterLondonSyntheticPopulation/` (or disable) |
| `--limit N` | Process only the first N areas (smoke tests) |

Per-area working outputs stay under `code/outputs/` as before. With collection enabled, primary tensors are also copied to:

```
GreaterLondonSyntheticPopulation/<area>/
    <area>_person_nodes.pt
    <area>_household_nodes.pt
    <area>_final_assignments.pt
manifest.csv
```

Assignment completion is detected via `outputs/assignment_hp_tuning_<area>/final_assignments.pt` (this project’s assignment output folder naming).
