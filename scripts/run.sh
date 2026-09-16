#!/bin/bash

DATA_ROOT=../datasets
PROJECT_ROOT=./
output_dir=${PROJECT_ROOT}/output/all
SEED=1

# Note: Parameters that may need to be passed: start, end, split, output_dir
flag="--root_dir ${DATA_ROOT} \           # Dataset root directory
      --img_root ../datasets/RGB_Observations_72_144 \  # RGB image observation data directory
      --split json_files/hinav_fail/MapGPT_72_scenes_processed_1.json \  # Dataset split name to use
      --start 0 \                         # Dataset sample start index (inclusive)
      --end 1 \                           # Dataset sample end index (exclusive)
      --output_dir ${output_dir} \         # Experiment results output directory
      --max_action_len 22 \                # Maximum action steps allowed per instruction
      --save_pred \                        # Save predicted navigation paths
      --stop_after 3 \                     # Minimum (non-stop) steps before allowing LLM to choose 'stop' action
      --llm gpt-4o \                       # Main LLM for environment understanding and planning
      --response_format json \             # Expected LLM response format as JSON
      --max_tokens 1000 \                  # Maximum tokens for LLM response generation
      --dataset r2r \                      # Dataset type (Room-to-Room)
      --model_name gpt-4o \                # Model identifier for logging/result naming
      --feedback_method gpt \              # Feedback method (generated using GPT)
      --eval_type val_unseen \             # Evaluation type (validation set unseen scenes)
      --action_level 1 \                   # Action abstraction level
      --action_gpt_model gpt-4o \          # LLM for generating specific action instructions
      --max_prompt_token 60000 \           # Maximum tokens for prompts sent to LLM
      --temperature 0.7 \                  # Controls LLM generation randomness (0.7 = moderate randomness)
      --error_mode error \                 # Error handling mode when encountering errors
      --num_beams 1 \                      # Number of beam search (1 = not using)
      --seed $SEED \                       # Random seed for reproducibility
      --enable_map_pruning \               # Enable dynamic map pruning
      --map_pruning_step_threshold 10 \    # Step threshold for map nodes to be considered 'old'
      --pruning_keep_recent_steps 3 \      # Step window to force keep recently visited nodes
      --pruning_start_step 15 \            # Step to start executing pruning
      --pruning_max_nodes_per_step 1 \     # Maximum nodes to prune per step
      --w_time 1.0 \                       # Time weight
      --w_degree 2.0 \                     # Degree weight
      --w_frontier 5.0 \                   # Frontier weight
      --w_dist 0.5 \                       # Distance weight
      --log_pruning_scores \               # Enable detailed scoring logging
      --extended_instruction \              # Enable extended instructions
      --extended_instr_file json_files/MapGPT_72_scenes_processed_sr.json \  # Extended instruction file
      $@"


python vln/main_gpt.py $flag
