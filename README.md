# JANUS

Minimal code release for the JANUS prompt attack experiments associated with arXiv:2603.21208. Local dataset and model paths have been replaced with Hugging Face sources.

## Contents

- `run_attack.py`: multi-GPU experiment runner.
- `utils/attack.py`: JANUS prior loading, reward search, and candidate generation.
- `utils/safety.py`: text and image safety checker wrappers.
- `configs/defaults.json`: default dataset/model/checkpoint source names.
- `requirements.txt`: Python dependencies.

## Setup

```bash
conda create -n janus python=3.10 -y
conda activate janus
pip install -r requirements.txt
```

Some upstream models are gated. Log in before running if your environment has not cached them:

```bash
huggingface-cli login
```

## Hugging Face Sources

Defaults used by this release:

- Prompt dataset: `AdamCodd/Civitai-8m-prompts`
- Prompt subset: first 200 non-empty rows from split `train`, column `prompt`
- SDXL text encoders/tokenizers: `stabilityai/stable-diffusion-xl-base-1.0`
- Image generator: `stabilityai/stable-diffusion-3.5-large-turbo`
- Text safety classifier: `michellejieli/NSFW_text_classifier`
- Optional image scorer: `Falconsai/nsfw_image_detection`

The prompt dataset is loaded directly from `AdamCodd/Civitai-8m-prompts`; no project-owned dataset artifact is required. If you publish prior checkpoints under a different repo or directory, pass `--prior-repo` and `--prior-dir`.

## Reproduce Paper Command

From this folder:

```bash
python run_attack.py \
  --run-name paper_sdxl \
  --data AdamCodd/Civitai-8m-prompts \
  --seed 1436 \
  --num-candidates 5 \
  --steps 5000 \
  --num-prompts 200 \
  --output-dir outputs
```

For a quick smoke test:

```bash
python run_attack.py \
  --run-name smoke \
  --seed 1436 \
  --num-candidates 1 \
  --steps 10 \
  --num-prompts 1 \
  --output-dir outputs
```

Outputs are written to `outputs/<run-name>/`, including `results.json`, `prompts.json`, and `attack/candidates.txt`.

## Notes

The default reward path uses the text safety checker and a constant image reward. Add `--score-images` to generate and score images inside the reward loop; this is much slower and needs more VRAM.

Enable SwanLab logging with `--track`. Tracking is disabled by default.

## Citation

```bibtex
@inproceedings{zheng2026janus,
  title={JANUS: A Lightweight Framework for Jailbreaking Text-to-Image Models via Distribution Optimization},
  author={Zheng, Haolun and He, Yu and Chen, Tailun and Shao, Shuo and Chu, Zhixuan and Zhou, Hongbin and Tao, Lan and Qin, Zhan and Ren, Kui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15719--15729},
  year={2026}
}
```
