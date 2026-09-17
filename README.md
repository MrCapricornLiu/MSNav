# 🚗 MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2508.16654-b31b1b.svg)](https://arxiv.org/abs/2508.16654)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Official implementation of **MSNav**, a zero-shot vision-and-language navigation framework combining dynamic map memory, spatial reasoning, and LLM-based action planning.

Project and repository led and maintained by [Chenghao Liu](https://github.com/MrCapricornLiu).

Our paper has been accepted by **ICASSP 2026** 🎉. Read it on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11463005) or [arXiv](https://arxiv.org/abs/2508.16654).

## 🔎 Overview

MSNav brings together three complementary modules for long-horizon navigation:

- **Memory:** Maintains a topological map and selectively prunes historical nodes to retain useful navigation context.
- **Spatial:** Uses Qwen-Spatial (Qwen-Sp), fine-tuned from Qwen3-4B, to infer relevant objects and destination layouts.
- **Decision:** Combines visual observations, navigation history, map memory, and spatial cues for GPT-based action planning.

> **Note:** This repository contains navigation, spatial inference, and evaluation code. Datasets, observation images, and fine-tuned checkpoints must be prepared separately.

## 📦 Repository Structure

```text
MSNav/
├── GPT/
│   ├── api.py                       # Vision-language API client
│   └── one_stage_prompt_manager.py  # Navigation prompts and spatial cues
├── Spatial/scripts/
│   ├── infer_instr_obj.py           # Instruction-to-object inference
│   ├── infer_instr_sr.py            # Destination spatial reasoning
│   └── eval_obj_metrics.py          # Object extraction evaluation
├── vln/
│   ├── main_gpt.py                  # Navigation evaluation entry point
│   ├── gpt_agent.py                 # Navigation agent and map pruning
│   ├── env.py                       # Matterport3D navigation environment
│   └── parser.py                    # Command-line configuration
├── utils/                          # Data loading and logging
├── scripts/run.sh                  # Experiment configuration reference
├── figs/placeholder_pruned.png      # Placeholder for pruned observations
└── requirements.txt                # Core Python dependencies
```

## ⚙️ Setup

### 1. Install Dependencies

```bash
git clone https://github.com/MrCapricornLiu/MSNav.git
cd MSNav
pip install -r requirements.txt
pip install h5py
```

**Additional requirements:**

- **Navigation:** Matterport3D Simulator with `MatterSim` Python bindings installed in the same environment.
- **Spatial inference:** PyTorch, Transformers, ModelScope Swift (`ms-swift`), and their dependencies.

The pinned `requirements.txt` covers core navigation dependencies, not the simulator or spatial-model stack.

### 2. Prepare Data

Prepare R2R connectivity graphs, Matterport3D scan data, processed navigation annotations, and RGB observations. The navigation code expects the following layout:

```text
DATA_ROOT/
├── R2R/
│   ├── connectivity/
│   └── annotations/
└── Matterport3D/
    └── v1_unzip_scans/

IMG_ROOT/
└── <scan_id>/<viewpoint_id>/<view_index>.jpg
```

Use `--root_dir` for `DATA_ROOT`, `--img_root` for `IMG_ROOT`, and `--split` for the processed annotation JSON. The example below uses a filename containing `processed` to select the processed-data loading branch.

### 3. Configure Local Paths

- [`GPT/api.py`](GPT/api.py): Set `generation_key` and the API `base_url`.
- [`vln/gpt_agent.py`](vln/gpt_agent.py): Point `self.placeholder_image_data` to the included `figs/placeholder_pruned.png`.
- [`Spatial/scripts/`](Spatial/scripts/): Set model checkpoints and input/output paths before spatial inference.

Keep API credentials local; do not commit them to the repository.

## 🧪 Navigation Evaluation

### Run Navigation

From the repository root, run the following command after completing setup. This example evaluates one instruction with map pruning enabled:

```bash
python -m vln.main_gpt \
    --root_dir /path/to/datasets \
    --img_root /path/to/RGB_observations \
    --split /path/to/MapGPT_72_scenes_processed_1.json \
    --start 0 \
    --end 1 \
    --output_dir output/msnav \
    --dataset r2r \
    --batch_size 1 \
    --llm gpt-4o \
    --response_format json \
    --max_action_len 22 \
    --max_tokens 1000 \
    --save_pred \
    --enable_map_pruning
```

> **Important:** `--start` is inclusive and `--end` is exclusive. Supply an explicit, valid `--end` for the current processed-data loader, and keep `--batch_size 1`.

### Add Spatial Cues

Append these arguments to the navigation command, continuing the preceding line with `\`:

```bash
    --extended_instruction \
    --extended_instr_file /path/to/extended_instructions.json
```

The extended JSON must contain matching `scan`, `path_id`, and `instruction` fields, together with `final_destination_spatial_relations`.

### Evaluation Outputs

Outputs are saved under `--output_dir`:

- **Trajectories and per-instruction metrics:** `preds/case_InstrID_*.json` (requires `--save_pred`).
- **Aggregate navigation metrics:** `logs/valid.txt`.

### Map Memory Configuration

| Argument | Purpose |
| --- | --- |
| `--enable_map_pruning` | Enable dynamic map pruning. |
| `--pruning_start_step` | Set the first step at which pruning is considered. |
| `--map_pruning_step_threshold` | Set the minimum age for candidate nodes. |
| `--pruning_keep_recent_steps` | Protect recently visited nodes. |
| `--pruning_max_nodes_per_step` | Limit the number of nodes pruned per step. |
| `--w_time`, `--w_degree`, `--w_frontier` | Weight node age, connectivity, and unexplored neighbors. |
| `--enable_graph_distance_pruning`, `--w_dist` | Enable and weight the graph-distance component. |
| `--log_pruning_scores` | Log candidate scores for inspection. |

See [`vln/parser.py`](vln/parser.py) for defaults and additional options. The API client currently uses a fixed temperature of `0`; there is no `--temperature` command-line option.

## 🧩 Spatial Reasoning

The spatial scripts use Qwen3-4B and configurable local checkpoints to extract objects and infer destination layouts.

| Script | Purpose |
| --- | --- |
| [`infer_instr_obj.py`](Spatial/scripts/infer_instr_obj.py) | Extract `direct_obj` and `potential_obj` lists from navigation instructions. |
| [`infer_instr_sr.py`](Spatial/scripts/infer_instr_sr.py) | Generate destination descriptions and spatial-layout cues. |
| [`eval_obj_metrics.py`](Spatial/scripts/eval_obj_metrics.py) | Evaluate object extraction with F1, NDCG, and weighted metrics. |

After configuring the checkpoint and data paths in the inference scripts:

```bash
python Spatial/scripts/infer_instr_obj.py
python Spatial/scripts/infer_instr_sr.py
```

The evaluation script retains local experiment configuration; review its model-loading and output-path settings before use. Qwen-Sp weights and I-O-S data are not bundled with the code.

## 📖 Citation

If you use MSNav in your research, please cite our paper:

```bibtex
@inproceedings{liu2026msnav,
  title     = {{MSNav}: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and {LLM} Spatial Reasoning},
  author    = {Liu, Chenghao and Zhou, Zhimu and Zhang, Jiachen and Zhang, Minghao and Huang, Songfang and Duan, Huiling},
  booktitle = {2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/abstract/document/11463005}
}
```

## License

This project is licensed under the [MIT License](./LICENSE).

## 🙏 Acknowledgements

We thank the authors of [NavGPT](https://github.com/GengzeZhou/NavGPT), [MapGPT](https://github.com/chen-judge/MapGPT), and [InstructNav](https://github.com/LYX0501/InstructNav) for their pioneering work in language-guided navigation. Their research and open-source contributions provide valuable foundations and inspiration for MSNav. We sincerely appreciate their efforts to advance the community.
