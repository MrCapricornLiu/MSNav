# MSNav

A Vision-and-Language Navigation (VLN) system that leverages GPT models for intelligent navigation with dynamic map pruning and adaptive path planning.

## Overview

MSNav is an advanced navigation agent that combines vision-language understanding with dynamic environment mapping. The system uses GPT-4o to interpret navigation instructions and make intelligent decisions about movement through complex environments.

### Key Features

- **GPT-powered Navigation**: Uses GPT-4o for environment understanding and path planning
- **Dynamic Map Pruning**: Intelligently prunes navigation maps to maintain efficiency
- **Multi-modal Input**: Processes both visual observations and textual instructions
- **Configurable Parameters**: Extensive customization options for different scenarios

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MSNav
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure GPT API:
   - Edit `GPT/api.py` and set your OpenAI API key and base URL
   - Replace `"xxx"` placeholders with your actual credentials


## Usage

### Dataset Support

The system supports Room-to-Room (R2R) navigation datasets. 

### Basic Usage

Run the navigation system using the provided script:

```bash
bash scripts/run.sh
```

### Custom Configuration

The system supports extensive configuration through command-line arguments:

```bash
python vln/main_gpt.py \
    --root_dir /your/dataset/root/location \
    --img_root /your/dataset/location \
    --split MapGPT_72_scenes_processed_1.json \
    --start 0 \
    --end 1 \
    --output_dir /your/output/location \
    --llm gpt-4o \
    --enable_map_pruning \
    --extended_instruction
```

### Key Parameters

- `--llm`: GPT model to use (e.g., gpt-4o)
- `--max_action_len`: Maximum number of actions per instruction
- `--enable_map_pruning`: Enable dynamic map pruning for efficiency
- `--temperature`: Control randomness in GPT responses (0.0-1.0)
- `--extended_instruction`: Use extended instruction format

## Project Structure

```
MSNav/
├── GPT/                    # GPT API and prompt management
│   ├── api.py             # OpenAI API integration
│   └── one_stage_prompt_manager.py
├── Spatial/               # Spatial reasoning and object analysis module
│   └── scripts/           # Processing scripts for spatial understanding
│       ├── eval_obj_metrics.py    # Object detection metrics evaluation
│       ├── infer_instr_obj.py        # Object inference from instructions
│       ├── infer_instr_sr.py         # Spatial layout reasoning
├── vln/                   # Core navigation logic
│   ├── main_gpt.py       # Main entry point
│   ├── gpt_agent.py      # GPT-powered navigation agent
│   ├── env.py            # Environment interface
│   └── ...
├── utils/                 # Utility functions
├── scripts/               # Execution scripts
└── requirements.txt       # Python dependencies
```

## Configuration

### Map Pruning Parameters

- `--map_pruning_step_threshold`: Steps before a node is considered 'old'
- `--pruning_keep_recent_steps`: Window of recent steps to preserve
- `--pruning_start_step`: When to start pruning
- `--pruning_max_nodes_per_step`: Maximum nodes to prune per step

### Scoring Weights

- `--w_time`: Time weight in scoring
- `--w_degree`: Degree weight in scoring  
- `--w_frontier`: Frontier weight in scoring
- `--w_dist`: Distance weight in scoring

## Spatial Module

The Spatial module provides advanced spatial reasoning and object analysis capabilities for Vision-and-Language Navigation tasks. It uses fine-tuned Qwen3-4B to understand spatial relationships and identify relevant objects from navigation instructions.

### Key Components

- **Object Inference (`infer_instr_obj.py`)**: You can use finetuned models to analyze navigation instructions to extract:
  - `direct_obj`: Objects explicitly mentioned in instructions (sorted by importance)
  - `potential_obj`: Other relevant objects that might be encountered (sorted by relevance)

- **Spatial Layout Reasoning (`infer_instr_sr.py`)**: You can use finetuned models to infer spatial layout of destinations by generating:
  - Destination descriptions based on instructions
  - Spatial arrangement of key objects at the destination

- **Evaluation Metrics (`eval_obj_metrics.py`)**: You can compute comprehensive metrics including:
  - Direct object F1 score
  - Potential object F1 score
  - Total F1 score
  - NDCG (Normalized Discounted Cumulative Gain)
  - Weighted scoring


### Usage

The Spatial module integrates with the main navigation system to provide enhanced spatial understanding. Configure your model checkpoints and API keys in the respective script files before running.



## Requirements

- Python 3.7+
- OpenAI API access
- Required packages listed in `requirements.txt`
