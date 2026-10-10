# Formal Closed-Set Label-Aware Drift Experiments

This directory contains feature-aligned SADA and label-aware detector results using strict task-aware option scoring.

## Protocol

- Window size: 200
- Windows per ratio: 1000
- Seeds: [42, 43, 44]
- Target calibration FPR: 5.0%
- Detector selection: maximize held-out drift validation power subject to the FPR constraint.
- OPTWIN is explicitly a local OPTWIN-style reproduction; River official implementations are used for the other methods.

## Calibration Status

| Setting | Method | Status |
|---|---|---|
| natural/binary_error | DDM | selected |
| natural/binary_error | EDDM | selected |
| natural/binary_error | ADWIN | selected |
| natural/binary_error | HDDM-W | selected |
| natural/binary_error | Page-Hinkley | selected |
| natural/binary_error | KSWIN | selected |
| natural/binary_error | OPTWIN | not_calibratable |
| natural/correct_option_nll | ADWIN | selected |
| natural/correct_option_nll | Page-Hinkley | selected |
| natural/correct_option_nll | KSWIN | selected |
| natural/correct_option_nll | OPTWIN | not_calibratable |
| task_balanced/binary_error | DDM | selected |
| task_balanced/binary_error | EDDM | selected |
| task_balanced/binary_error | ADWIN | selected |
| task_balanced/binary_error | HDDM-W | selected |
| task_balanced/binary_error | Page-Hinkley | selected |
| task_balanced/binary_error | KSWIN | selected |
| task_balanced/binary_error | OPTWIN | selected |
| task_balanced/correct_option_nll | ADWIN | selected |
| task_balanced/correct_option_nll | Page-Hinkley | selected |
| task_balanced/correct_option_nll | KSWIN | selected |
| task_balanced/correct_option_nll | OPTWIN | not_calibratable |

Machine-readable runs and summaries are stored in the adjacent JSON files.
