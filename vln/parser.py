import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r')
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # Data preparation
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)  # only support bach_size=1

    # --- Unified Map Pruning Parameters (V7 Strategy) ---
    parser.add_argument('--enable_map_pruning', action='store_true', default=False, 
                        help='Master switch to enable the map pruning strategy.')
    
    # Controls when pruning starts
    parser.add_argument('--pruning_start_step', type=int, default=15, 
                        help='Step number from which to start the pruning logic.')

    # Basic temporal filtering for candidate eligibility
    parser.add_argument('--map_pruning_step_threshold', type=int, default=13, 
                        help='Node is considered \'old enough\' if steps_since_last_visit > this threshold.')
    parser.add_argument('--pruning_keep_recent_steps', type=int, default=6, 
                        help='Node is considered \'not recent\' if steps_since_last_visit > this threshold (Force keeps nodes visited within this window).')

    # Scoring and selection controls for eligible candidates
    parser.add_argument('--pruning_max_nodes_per_step', type=int, default=1, 
                        help='Maximum number of eligible candidate nodes to prune in a single step.')
    parser.add_argument('--w_time', type=float, default=1.0, 
                        help='Weight for the time factor (steps_since_last_visit) in pruning score.')
    parser.add_argument('--w_degree', type=float, default=2.5, 
                        help='Weight for the negative node degree factor in pruning score.')
    parser.add_argument('--w_frontier', type=float, default=5.0, 
                        help='Weight for the negative number of unexplored neighbors factor (frontier penalty) in pruning score.')
    parser.add_argument('--enable_graph_distance_pruning', action='store_true', default=False, 
                        help='Enable graph distance calculation and its use in pruning score.')
    parser.add_argument('--w_dist', type=float, default=0.5, 
                        help='Weight for the graph distance factor in pruning score (only effective if graph distance is enabled).')

    parser.add_argument('--log_pruning_scores', action='store_true', default=False,
                        help='Log detailed scoring components for pruning candidates.')

    parser.add_argument('--use_beam_search', action='store_true', default=False)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)
    parser.add_argument("--save_pred", action='store_true', default=False)

    # LLM
    parser.add_argument('--llm', type=str, default='')
    parser.add_argument('--response_format', type=str, default='str', choices=['str', 'json'])
    parser.add_argument('--img_root', type=str, default=None)
    parser.add_argument("--split", type=str, default='MapGPT_72_scenes_processed')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--stop_after', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=1000)

    # Spatial Reasoning
    parser.add_argument('--extended_instruction', action='store_true', default=False)
    parser.add_argument('--extended_instr_file', type=str, default='xxx.json')

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    if args.dataset == 'r2r':
        args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    elif args.dataset == 'reverie':
        args.anno_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')
    args.vis_dir = os.path.join(args.output_dir, 'vis')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    return args

