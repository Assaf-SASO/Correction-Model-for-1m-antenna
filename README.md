# Frequency Correction Model

This Python code is part of a research study investigating how the calibration results of electric field probes vary depending on the separation distance between the transmitting antenna and the probe.

## Research Background

The study compares a standard 1.5 m configuration (recommended by IEEE Std 1309) with a shorter 1.0 m configuration. Measurements taken at 1.0 m consistently under-report the actual field strength compared to those taken at 1.5 m. This systematic difference is quantified as a correction factor (ΔCF), which varies with frequency.

## Purpose

The script automates the entire calibration correction process:

1. Fits polynomial models of varying degrees (2-5) to actual ΔCF data
2. Selects the best model based on statistical accuracy (lowest mean squared error)
3. Uses the best-fit model to predict the appropriate ΔCF at any frequency
4. Adjusts real 1.0 m correction factor measurements to produce corrected values that approximate standard 1.5 m results

## Outputs

The code generates:
- Plots visualizing the polynomial models and their fit to the data
- Data files summarizing model performance
- Predicted ΔCF values at reference frequencies
- Final corrected calibration results

All outputs are saved in a timestamped directory for traceability.

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the model:
   ```
   python correction_model_for_1m_antenna.py
   ```

3. Examine the output files in the generated timestamped directory (`output_YYYYMMDD_HHMMSS`)

## Functions

- `predict_delta_cf(frequency)`: Predicts the ΔCF value for a given frequency using the trained model
- `apply_correction(frequency, measured_cf_1m)`: Applies correction to a 1.0 m measurement based on the ΔCF model

This code operationalizes the correction method proposed in the research — it's the bridge between experimental measurements and the traceability restoration model discussed in the results and discussion sections.
