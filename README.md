# SmKS Processing Example

This directory contains a small Python workflow for measuring and post-processing
SmKS-related seismic waveform residuals. The main script is `Main_preprocess.py`,
and most reusable logic lives in `smks_utils/`.

## Directory Structure

```text
.
├── Main_preprocess.py          # Main workflow script
├── smks_env.yml                # Conda environment export
├── smks_utils/                 # Project helper functions
│   ├── data_utils.py           # Data loading, merging, filtering, resampling
│   ├── measure_smks_utils.py   # Residual measurement and correlation tools
│   ├── raypath_utils.py        # Ray path and geodesy helpers
│   ├── selection_utils.py      # Auto-selection, clustering, group summaries
│   └── visualization_utils.py  # Plotting utilities
└── data/
    ├── df_all_groups.csv       # Combined group output table
    └── YYYYMMDD_HHMM/          # One folder per event
        ├── hypo.csv            # Event hypocenter metadata
        ├── df_stations_merge6.csv
        ├── np_waveforms_*.npy  # Waveform arrays
        ├── df_measured.csv     # Measurement output
        ├── df_selected.csv     # Auto-selection output
        ├── df_clustered.csv    # Cluster output
        ├── df_groups.csv       # Group summary output
        └── Figures/            # Generated figures
```

## Install the Environment

The provided `smks_env.yml` is an exported Conda environment. It is named `base`
inside the file, so create it with a new name to avoid modifying your own base
environment:

```bash
conda env create --file smks_env.yml --name smks
conda activate smks
```

## Run the Main Process

From this directory, run:

```bash
python3 Main_preprocess.py
```

The first section measures residuals and writes `df_measured.csv` plus figures
inside each event folder. The post-processing section performs auto-selection, trace clustering, group summarization, and writes updated CSV outputs under `data/`.

## Typical Workflow

1. Put each event in its own `data/YYYYMMDD_HHMM/` folder.
2. Make sure the event folder contains `hypo.csv`, `df_stations_merge6.csv`,
   and the required `np_waveforms_?.npy` files. The column `tridx` in 	`df_stations_merge6.csv` is consistent with the order in `np_waveforms_?.py`.
   From 0 to 5 are: 
   - observed
   - simulated on PREM 
   - simulated on PREM + MANTLE(SP12RTS) 
   - simulated on PREM + MANTLE(SP12RTS) + CORE(Torus)
   - simulated on PREM + MANTLE(SP12RTS) + CORE(Global)
   - simulated on PREM + MANTLE(SP12RTS) + CORE(subLLSVP) 
3. Activate the environment with `conda activate smks`.
4. Run `python Main_preprocess.py`.
5. Check the generated CSV files and plots in each event folder.

