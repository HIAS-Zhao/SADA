# SADA

Code and paper-result snapshots for **SADA: drift-aware resource allocation and adaptation**.

We keep our final-paper experiments in the following reproducibility directories:

- `drift_detection/` contains the final detector, ablation, held-out sensitivity, early-recall, and overhead experiments. The released paper configuration is `V+VT`, PCA 128, window 200, and calibration quantile 0.99.
- `framework/` contains the final three-seed framework protocol (seeds 42, 43, and 44), the published group-level/aggregate tables, protocol snapshots, and the source modules needed to materialize the original experiment layout.
- `figures/` contains the paper's final resource-allocation and smoothed ship-case analysis inputs and builders.

The CSV and Markdown files under `paper_results/` are the released paper evidence.

Installation profiles and environment requirements are described in [Environment setup](docs/environment.md). `requirements.txt` installs the CPU analysis dependencies; detector, model, and optional evaluation dependencies have separate profiles.

## Quick checks

```bash
python framework/scripts/verify_paper_results.py
python -m compileall drift_detection/src drift_detection/scripts framework/scripts
```

`framework/scripts/prepare_workspace.py` expands the archived `@@WORKSPACE@@` paths into a user-selected experiment workspace. It refuses to overwrite a file that differs from the released source.

## Datasets and models

We evaluate SADA on two dataset components:

- **Our custom dataset**: a multi-task, multimodal remote-sensing dataset assembled for the drift-detection and adaptation experiments. [Hugging Face repository](https://huggingface.co/datasets/qifeng24/SADA).
- **The public OmniEarth dataset**: available from [Hugging Face](https://huggingface.co/datasets/sjeeudd/OmniEarth), with documentation in the [official repository](https://github.com/one87624/OmniEarth).

We use Qwen2.5-VL and Qwen3.5 models from the [official Qwen model collection](https://huggingface.co/Qwen), RemoteCLIP-RN50 and RemoteCLIP-ViT-B/32 from the [RemoteCLIP project](https://github.com/om-ai-lab/RemoteCLIP), and ResNet18 and ResNet50 from [Torchvision](https://pytorch.org/vision/stable/models.html). Follow the corresponding project documentation to download the original model weights.

We use [MS-Swift](https://github.com/modelscope/ms-swift) for Qwen training and the upstream RemoteCLIP implementation with matching weights for RemoteCLIP evaluation.

## License

Code is released under the MIT License. Results and source-data-derived tables remain subject to the licenses of their underlying datasets and pretrained models.
