# NN4Psych Analysis Pipeline Guide

**Last Updated:** 2025-11-17
**Version:** v1
**Project:** Neural Networks for Psychological Modeling

---

## Overview

This pipeline processes trained RNN actor-critic models to extract and analyze behavioral data from predictive inference tasks. The pipeline follows standardized data analysis project conventions for reproducibility and maintainability.

---

## Pipeline Architecture

```
Stage 01: Extract Behavior → Stage 02: Compute Metrics → Stage 03: Analyze Hyperparameters
         ↓                            ↓                            ↓
    Trained Models             Learning Rates              Parameter Sweeps
    (.pth files)              & Summary Stats            & Performance Trends
```

---

## Stage 01: Behavior Extraction

### Script
`scripts/data_pipeline/01_extract_model_behavior.py`

### Description
Loads trained model weights and runs them through the predictive inference task environments (change-point and oddball conditions) to extract behavioral data.

### Input
- Model weight files (`.pth`) from `model_params*/` directories
- Model architecture defined in `config.py`

### Output
- `output/behavioral_summary/task_trials_long.csv` - Long-format trial data
- `output/behavioral_summary/raw_behavior_data.pickle` - Raw state vectors

### Columns Created
- `model_id` - Unique model identifier
- `epoch` - Training epoch number
- `trial` - Trial number within epoch
- `condition` - Task condition (change-point or oddball)
- `bucket_position` - Agent's predicted position
- `bag_position` - True bag drop position
- `helicopter_position` - Hidden state position
- `hazard_trigger` - Binary indicator of hazard event
- `prediction_error` - Signed error (bag - bucket)
- `abs_prediction_error` - Absolute prediction error

### Run
```bash
cd /path/to/nn4psych
python scripts/data_pipeline/01_extract_model_behavior.py
```

---

## Stage 02: Compute Learning Metrics

### Script
`scripts/data_pipeline/02_compute_learning_metrics.py`

### Description
Processes trial data to compute learning rates, prediction errors, and aggregated performance metrics.

### Input
- `output/behavioral_summary/task_trials_long.csv`

### Output
- `output/behavioral_summary/learning_rates_by_condition.csv` - Learning rate observations
- `output/behavioral_summary/summary_performance_metrics.csv` - Aggregated metrics per model

### Key Computations
1. **Learning Rate**: `update / prediction_error` where:
   - `update = diff(bucket_position)`
   - Only computed when `abs_prediction_error > 20`
   - Clipped to [0, 1] range

2. **Summary Statistics**:
   - Mean, std, median of prediction errors
   - Mean, std, median of learning rates
   - Hazard event frequency

### Run
```bash
python scripts/data_pipeline/02_compute_learning_metrics.py
```

---

## Stage 03: Hyperparameter Analysis

### Script
`scripts/data_pipeline/03_analyze_hyperparameter_sweeps.py`

### Description
Analyzes model performance across different hyperparameter configurations by parsing filenames and aggregating results.

### Input
- Model files with encoded hyperparameters in filenames
- Expected format: `{perf}_{version}_{gamma}g_{preset}rm_{rollout}bz_...`

### Output
- `output/parameter_exploration/gamma_sweep_results.csv`
- `output/parameter_exploration/rollout_sweep_results.csv`
- `output/parameter_exploration/preset_sweep_results.csv`
- `output/parameter_exploration/scale_sweep_results.csv`

### Sweep Values
- **Gamma (discount)**: [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
- **Rollout (batch size)**: [5, 10, 20, 30, 50, 75, 100, 150, 200]
- **Preset (memory init)**: [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
- **Scale (TD scale)**: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

### Run
```bash
python scripts/data_pipeline/03_analyze_hyperparameter_sweeps.py
```

---

## Master Pipeline Runner

### Script
`scripts/data_pipeline/00_run_full_pipeline.py`

### Description
Executes all pipeline stages in sequence with error handling and stage selection.

### Usage
```bash
# Run entire pipeline
python scripts/data_pipeline/00_run_full_pipeline.py

# Start from specific stage
python scripts/data_pipeline/00_run_full_pipeline.py --start 2

# Run specific range
python scripts/data_pipeline/00_run_full_pipeline.py --start 2 --end 3

# List all stages
python scripts/data_pipeline/00_run_full_pipeline.py --list
```

---

## Analysis Scripts

Located in `scripts/analysis/`:

### visualize_learning_rates.py
Creates behavioral analysis figures:
- Learning rate by prediction error curves
- Learning rate distribution by condition
- Model comparison summaries

**Output**: `figures/behavioral_summary/`

### train_example.py
Example training script demonstrating package usage with configuration system.

### analyze_hyperparams_unified.py
Unified hyperparameter analysis (replaces 5 duplicate scripts).

---

## Data Integration Pattern

All computed metrics are integrated into main data files rather than creating separate CSVs:

```python
# Example: Adding new metrics to existing data
df_trials = pd.read_csv(TRIALS_DATA_PATH)

# Compute new metric
df_trials['new_metric'] = compute_metric(df_trials)

# Save back to same file
df_trials.to_csv(TRIALS_DATA_PATH, index=False)
```

**Benefits**:
- Single source of truth
- No merging in downstream scripts
- Maintains data lineage

---

## Configuration

All paths and parameters are centralized in `config.py`:

```python
from config import (
    OUTPUT_DIR,
    BEHAVIORAL_SUMMARY_DIR,
    MODEL_PARAMS,
    TASK_PARAMS,
    COLUMN_NAMES,
)
```

**Key Configuration Sections**:
- `MODEL_PARAMS` - Actor-Critic architecture settings
- `TASK_PARAMS` - Environment configuration
- `TRAINING_PARAMS` - Training hyperparameters
- `COLUMN_NAMES` - Standardized column naming
- `*_VALUES` - Hyperparameter sweep ranges

---

## Output Directory Structure

```
output/
├── behavioral_summary/
│   ├── task_trials_long.csv
│   ├── learning_rates_by_condition.csv
│   ├── summary_performance_metrics.csv
│   └── raw_behavior_data.pickle
├── model_performance/
│   ├── hyperparameter_sweep_results.csv
│   └── best_performing_models.csv
└── parameter_exploration/
    ├── gamma_sweep_results.csv
    ├── rollout_sweep_results.csv
    ├── preset_sweep_results.csv
    └── scale_sweep_results.csv

figures/
├── behavioral_summary/
│   ├── learning_rate_by_prediction_error.png
│   ├── learning_rate_distribution.png
│   └── model_comparison_summary.png
├── model_performance/
└── parameter_exploration/
```

---

## Validation

Tests are located in `validation/` (for numbered tests) and `tests/` (for pytest):

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=nn4psych
```

---

## Reproducibility Checklist

- [ ] Set random seed in `config.py` (`DEFAULT_SEED = 42`)
- [ ] Use configuration files rather than hard-coded values
- [ ] Document model versions in output filenames
- [ ] Save experiment configurations alongside results
- [ ] Use consistent column naming from `COLUMN_NAMES`

---

## Common Issues & Solutions

### Missing Model Directory
```
No model_params directories found
```
**Solution**: Train models first or download pre-trained weights.

### Empty Output Files
```
No data extracted
```
**Solution**: Check model file format and architecture compatibility.

### Import Errors
```
ModuleNotFoundError: No module named 'nn4psych'
```
**Solution**: Install package with `pip install -e .` from project root.

---

## Next Steps

1. **Add Bayesian Model Fitting**: Implement PyMC fitting in `scripts/fitting/`
2. **Parameter Recovery**: Add validation tests for parameter recovery
3. **Group Analysis**: Compare models by clinical/behavioral groups
4. **Neural Analysis**: Add fixed-point analysis pipeline

---

## References

- Nassar et al. (2021) - Change-point and oddball learning paradigms
- Wang et al. (2018) - RNN actor-critic foundations
- Data Analysis Project Template v2.0

---

**Template Compliance**: Follows QUICK_START_TEMPLATE.md conventions for data analysis projects.
