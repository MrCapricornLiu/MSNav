
import sys
import numpy as np
from collections import defaultdict, deque
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer
import json
import math
import time
from typing import List, Any


class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args
        self.env = env

        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        self.results = {}

        self.target_idx = 0
        self.node_last_visit_step = defaultdict(int)

        # Placeholder image for pruned nodes (V7) - Path to the image file
        self.placeholder_image_data = "xxx/placeholder_pruned.png"
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])

    def _calculate_graph_distances(self, graph, start_node):
        if start_node not in graph:
            return {} # Start node itself is not in the graph, no distances to calculate from it.
            
        distances = {node_id: float('inf') for node_id in graph}
        distances[start_node] = 0
        queue = deque([start_node])
        
        visited_bfs = {start_node}

        while queue:
            current_g_node = queue.popleft()
            
            # Neighbors are defined by graph[current_g_node] which is a list of viewpointIds
            for neighbor_id in graph.get(current_g_node, []):
                if neighbor_id in graph and neighbor_id not in visited_bfs: # Ensure neighbor is part of the current graph
                    visited_bfs.add(neighbor_id)
                    distances[neighbor_id] = distances[current_g_node] + 1
                    queue.append(neighbor_id)
                elif neighbor_id not in graph and distances.get(neighbor_id) == float('inf'):
                    # If a listed neighbor is NOT in the graph (e.g. already pruned), it remains inf distance
                    pass


        return distances

    def _prune_map(self, current_step):
        # Step 1: Trigger Conditions
        if not self.args.enable_map_pruning:
            return
        if current_step < self.args.pruning_start_step:
            return

        batch_idx = 0 # Assuming batch_size is 1 for pruning logic
        
        current_graph = self.prompt_manager.graph[batch_idx]
        if not current_graph: # Nothing to prune
            return
            
        trajectory = self.prompt_manager.trajectory[batch_idx]
        if not trajectory: # Should not happen if navigation has started
            return
        current_viewpoint_id = trajectory[-1]

        # Step 2: Candidate Identification & Feature Calculation
        pruning_candidates_with_scores = []

        # Calculate graph distances if enabled
        graph_distances = {}
        if self.args.enable_graph_distance_pruning:
            graph_distances = self._calculate_graph_distances(current_graph, current_viewpoint_id)

        for node_id in list(current_graph.keys()): # Iterate over a copy of keys for safe modification
            # Rule 1: Non-current
            if node_id == current_viewpoint_id:
                continue

            # Rule 2: Explored
            if node_id not in trajectory:
                continue
            
            steps_since_last_visit = current_step - self.node_last_visit_step.get(node_id, -1)

            # Rule 3: Non-recent
            if steps_since_last_visit <= self.args.pruning_keep_recent_steps:
                continue
            
            # Rule 4: Sufficiently Old
            if steps_since_last_visit <= self.args.map_pruning_step_threshold: # Note: V7 said ">", this should be consistent. If it is ">" it means if it equals threshold, it's not pruned.
                                                                            # Let's stick to ">" as "more than threshold" implies prunable.
                continue


            # Features for scoring
            degree = len(current_graph.get(node_id, []))
            
            num_unexplored_neighbors = 0
            for neighbor in current_graph.get(node_id, []):
                if neighbor not in trajectory:
                    num_unexplored_neighbors += 1
            
            node_graph_distance = float('inf') # Default if not reachable or not calculated
            if self.args.enable_graph_distance_pruning:
                 node_graph_distance = graph_distances.get(node_id, float('inf'))


            # Add to candidates list for scoring
            pruning_candidates_with_scores.append({
                'id': node_id,
                'steps_since_last_visit': steps_since_last_visit,
                'degree': degree,
                'num_unexplored_neighbors': num_unexplored_neighbors,
                'graph_distance': node_graph_distance
            })

        if not pruning_candidates_with_scores:
            return

        # Step 3: Candidate Scoring
        if self.args.log_pruning_scores and pruning_candidates_with_scores:
            print(f"--- Pruning Scores Detail (Step {current_step}, Current VP: {current_viewpoint_id}) ---")

        for candidate in pruning_candidates_with_scores:
            # Raw features
            raw_steps_since_last_visit = candidate['steps_since_last_visit']
            raw_degree = candidate['degree']
            raw_num_unexplored_neighbors = candidate['num_unexplored_neighbors']
            raw_graph_distance = candidate['graph_distance']

            # Score components
            # Adjusted Time Score (V7.1)
            effective_age = raw_steps_since_last_visit - self.args.map_pruning_step_threshold 
            time_contribution = self.args.w_time * effective_age
            
            degree_contribution = self.args.w_degree * (-raw_degree) 
            frontier_contribution = self.args.w_frontier * (-raw_num_unexplored_neighbors)
            
            dist_contribution = 0
            if self.args.enable_graph_distance_pruning and raw_graph_distance != float('inf'):
                dist_contribution = self.args.w_dist * raw_graph_distance
            
            candidate['pruning_score'] = time_contribution + degree_contribution + frontier_contribution + dist_contribution

            if self.args.log_pruning_scores:
                print(f"  Node {candidate['id']}:")
                print(f"    Raw: age={raw_steps_since_last_visit}, deg={raw_degree}, unexpl_neigh={raw_num_unexplored_neighbors}, dist={raw_graph_distance if raw_graph_distance != float('inf') else 'N/A'}")
                print(f"    Contrib: time={time_contribution:.2f} (eff_age={effective_age}), deg={degree_contribution:.2f}, front={frontier_contribution:.2f}, dist={dist_contribution:.2f}")
                print(f"    Total Score: {candidate['pruning_score']:.2f}")
        
        if self.args.log_pruning_scores and pruning_candidates_with_scores:
            print(f"-----------------------------------------------------")

        # Step 4: Pruning Selection
        pruning_candidates_with_scores.sort(key=lambda x: x['pruning_score'], reverse=True) # Highest score first
        
        nodes_to_prune_ids = [
            cand['id'] for cand in pruning_candidates_with_scores[:self.args.pruning_max_nodes_per_step]
        ]

        if not nodes_to_prune_ids:
            return

        # Step 5: Execution
        nodes_list_for_batch = self.prompt_manager.nodes_list[batch_idx]
        node_imgs_for_batch = self.prompt_manager.node_imgs[batch_idx]

        for node_id_to_prune in nodes_to_prune_ids:
            if node_id_to_prune in current_graph: # Double check it's still there
                # A. Update graph: Remove node and its edges
                del current_graph[node_id_to_prune]
                for graph_node_id in list(current_graph.keys()): # Iterate over copy
                    if node_id_to_prune in current_graph[graph_node_id]:
                        current_graph[graph_node_id].remove(node_id_to_prune)
                
                # B. Update access time
                if node_id_to_prune in self.node_last_visit_step:
                    del self.node_last_visit_step[node_id_to_prune]
                
                # C. Image Placeholder
                try:
                    # nodes_list maps viewpointId to its display index (Place 0, Place 1, etc.)
                    # The image for viewpointId X is at index X in node_imgs
                    idx_in_nodes_list = nodes_list_for_batch.index(node_id_to_prune)
                    if 0 <= idx_in_nodes_list < len(node_imgs_for_batch):
                         node_imgs_for_batch[idx_in_nodes_list] = self.placeholder_image_data
                except ValueError:
                    pass
        

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]
        self.node_last_visit_step.clear()
        # Initial step (t=0) state is handled by make_action_prompt adding the first node

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            # Update graph, trajectory and get candidate actions based on current obs
            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)

            # Assuming batch size 1
            batch_idx = 0
            if not self.prompt_manager.trajectory[batch_idx]: 
                 print(f"Warning: Trajectory for batch {batch_idx} is empty at step {t}. Skipping step.")
                 continue # Should not happen if make_action_prompt worked correctly
                 
            # Update last visit step for the current node (which was just added by make_action_prompt)
            current_vp = self.prompt_manager.trajectory[batch_idx][-1]
            self.node_last_visit_step[current_vp] = t 

            # Prune the map based on the updated graph and visit steps
            self._prune_map(t)

            # --- LLM Prompt Generation --- 
            if self.args.response_format == 'str':
                nav_input = self.prompt_manager.make_r2r_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            elif self.args.response_format == 'json':
                nav_input = self.prompt_manager.make_r2r_json_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            if self.args.llm == 'gpt-4-vision-preview' and self.args.response_format == 'str':
                # GPT-4V only supports string mode output
                nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                               self.args.llm, self.args.max_tokens)
                print('-------------------- Output --------------------')
                print(nav_output)
                print(tokens)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(nav_output=nav_output,
                                                       only_options_batch=nav_input["only_options"],
                                                       t=t)
                self.prompt_manager.parse_planning(nav_output=nav_output)

            # elif self.args.llm == 'gpt-4o-2024-05-13' and self.args.response_format == 'json':
            elif self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                if len(image_list) > 35:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                                   self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})

                    print(f"DEBUG: Received nav_output (type: {type(nav_output)}):")
                    print(f"'''\n{nav_output}\n'''") # Print the original output
                    print(tokens)
                    try:
                        json_output = json.loads(nav_output)
                        a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                        self.prompt_manager.parse_json_planning(json_output)
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Failed to decode JSON: {e}")
                        print("ERROR: Original nav_output was:", repr(nav_output))
                        # Take default behavior, for example, stop
                        print("WARN: JSONDecodeError occurred. Defaulting to action index 0 (stop).")
                        a_t = [0] # Assume index 0 is the stop action
                    # --- End of Debugging and Error Handling ---

                    print('-------------------- Output --------------------')
                    # If parsing succeeds, print the parsed result, otherwise it may be empty or undefined
                    if 'json_output' in locals():
                        print(json.dumps(json_output, indent=2)) # Print the formatted JSON
                    else:
                        print("Output: (JSON decoding failed)")
                    
            elif self.args.llm == 'gpt-4o-mini' and self.args.response_format == 'json':
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                                   self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                    json_output = json.loads(nav_output)
                    a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                    self.prompt_manager.parse_json_planning(json_output)
                    print('-------------------- Output --------------------')
                    print(nav_output)
                    print(tokens)
            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj
