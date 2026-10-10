# Environment setup

Use Python 3.10 for the experiment installation profiles. The recorded detector environment uses Python 3.10.19 on Linux, NumPy 2.2.5, SciPy 1.15.3, scikit-learn 1.7.2, and PyTorch 2.9.1; see `drift_detection/paper_results/performance_degrading_early_recall_20260920/early_recall_config.json`. These four package versions are retained in the requirements. The remaining direct dependency versions specify an installation profile; the experiment records do not contain a complete original package lock.

## Installation profiles

Run the commands from the repository root in a virtual environment. Each profile includes its shared requirements.

| Requirements file | Purpose |
|---|---|
| `requirements.txt` or `requirements/analysis.txt` | CPU numerical analysis, CSV summaries, PCA utilities, and plotting |
| `requirements/detectors.txt` | Detector experiments using cached PyTorch features and River comparisons |
| `requirements/vision.txt` | RemoteCLIP and ResNet feature extraction and evaluation |
| `requirements/qwen.txt` | Qwen2.5-VL and Qwen3.5 inference, feature extraction, LoRA training, and Fisher computation |
| `requirements/evaluation.txt` | Optional BLEU, ROUGE-L, METEOR, CIDEr, and API-based scoring |
| `requirements/all.txt` | All profiles |

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python framework/scripts/verify_paper_results.py
```

Published aggregate verification and the held-out CSV summary use the Python standard library and can run without installing model packages. Generating figures with the released builders additionally requires the external profiles and prediction bundles specified in `figures/`.

For detector runs with CPU features, install the CPU PyTorch build first:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/detectors.txt
```

For CUDA model experiments, choose the PyTorch wheel index supported by the host driver. For example, the CUDA 12.8 profile is:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/qwen.txt
python -m pip install -r requirements/vision.txt
python -m pip check
```

Training and framework execution require Linux because the released training entry points use `fcntl` and Linux accelerator paths. They also require the datasets, model weights, training manifests, runtime profiles, and checkpoints described in the component READMEs.

## Version compatibility

The PyPI package metadata supplies the following constraints for the installation profile:

- SciPy 1.15.3 accepts NumPy 1.23.5 through versions below 2.5; NumPy 2.2.5 satisfies this range.
- scikit-learn 1.7.2 requires Python 3.10 or newer, NumPy 1.22 or newer, SciPy 1.8 or newer, and joblib 1.2 or newer.
- torchvision 0.24.1 requires PyTorch 2.9.1.
- River 0.22.0 supports Python 3.10, SciPy 1.14.1 or newer, and pandas 2.2.3 through versions below 3.0. pandas is supplied as a River or MS-Swift dependency; the repository does not use it directly.
- MS-Swift 4.0.0 accepts Transformers versions below 5.3.0 and PEFT versions below 0.19. Transformers 5.2.0 and PEFT 0.18.1 meet these constraints. MS-Swift 4.0.0 contains the `SwiftSft`, `TrainerFactory`, and `per_token_loss_func` interfaces used by the training scripts, and Transformers 5.2.0 provides the Qwen2.5-VL and Qwen3.5 model implementations.

All directly declared packages are fixed to exact versions. Platform-specific transitive packages are resolved by pip; these files are installation profiles rather than a complete platform lock. Model and CUDA execution must be verified on the intended accelerator host.

## Optional components

The River audit scripts accept an explicit installation directory. A package installation can be selected with:

```bash
RIVER_PATH=$(python -c "import pathlib, river; print(pathlib.Path(river.__file__).resolve().parent.parent)")
python drift_detection/scripts/audit_label_aware_detector_protocol.py --river-path "$RIVER_PATH" --help
```

For caption and generation metrics:

```bash
python -m pip install -r requirements/evaluation.txt
python -m nltk.downloader wordnet omw-1.4
```

Install these metric packages when reproducing the corresponding scores. BLEU, ROUGE-L, and CIDEr utilities otherwise use approximations, which produce different metrics. API-based scoring additionally requires the configured endpoint, model access, and `OPENAI_API_KEY`.

FlashAttention is an optional compiled accelerator extension and is excluded from the portable profiles. The archived training commands select FlashAttention by default; either install the extension matching the chosen PyTorch/CUDA build or set the supported attention option to `sdpa`. GeoChat, MiniCPM, and InternVL branches in the generic evaluation utility require their upstream model code and model-specific environments; they are outside the paper's Qwen, RemoteCLIP, and ResNet installation profiles.
