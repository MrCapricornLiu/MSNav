import os
import json
import numpy as np
import argparse
import re  # Add re module for regular expression processing
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def compute_json_metrics(_, tokenizer=None, decoded_preds_and_labels=None):
    """
    Compute various metrics between predicted objects and label objects
    
    Args:
        tokenizer: Not used, for interface compatibility
        decoded_preds_and_labels: Tuple (all_preds, labels), predictions and labels
    
    Returns:
        Dictionary containing various metrics and detailed metrics for each sample
    """
    all_preds, labels = decoded_preds_and_labels
    
    # Initialize metrics
    metrics = {
        'direct_obj_f1': 0.0,
        'potential_obj_f1': 0.0,
        'total_f1': 0.0,
        'ndcg': 0.0,
        'weighted_score': 0.0
    }
    
    total_samples = len(all_preds)
    if total_samples == 0:
        return metrics, []
    
    # Used to store metrics for all samples, calculate average at the end
    direct_f1_scores = []
    potential_f1_scores = []
    total_f1_scores = []
    ndcg_scores = []
    weighted_scores = []
    
    # Store detailed metrics for each sample
    per_sample_metrics = []
    
    # Add a more lenient JSON parsing function
    def robust_json_parse(json_str):
        """
        Try to parse JSON string in a more lenient way
        Handle common JSON format issues like trailing commas
        """
        if not isinstance(json_str, str):
            return None
        
        # Clean JSON string
        # 1. Remove trailing commas
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # 2. Try to fix unclosed brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        # 3. Try to parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    for idx, (pred, label) in enumerate(zip(all_preds, labels)):
        try:
            def extract_json_from_text(text):
                if not isinstance(text, str) or not text.strip():
                    print(f"Warning: Invalid input, non-string or empty: {text}")
                    return {}
                if "<think>" in text and "</think>" in text:
                    think_start = text.find("<think>")
                    think_end = text.find("</think>", think_start) + len("</think>")
                    text = text[:think_start] + text[think_end:]
                
                text = text.replace("/no_think/", "").replace("/nothink/", "")
                
                text = text.strip()
                
                if "```json" in text and "```" in text.split("```json", 1)[1]:
                    json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
                    json_obj = robust_json_parse(json_str)
                    if json_obj:
                        return json_obj
                    print(f"Failed to parse JSON from ```json block")
                elif "```" in text and "```" in text.split("```", 1)[1]:
                    json_str = text.split("```", 1)[1].split("```", 1)[0].strip()
                    json_obj = robust_json_parse(json_str)
                    if json_obj:
                        return json_obj
                    print(f"Failed to parse JSON from normal code block")
                
                try:
                    start_idx = text.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        for i in range(start_idx, len(text)):
                            if text[i] == '{':
                                brace_count += 1
                            elif text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = text[start_idx:i+1]
                                    json_obj = robust_json_parse(json_str)
                                    if json_obj:
                                        return json_obj
                except Exception:
                    print(f"Failed to extract JSON object from text")
                
                json_obj = robust_json_parse(text)
                if json_obj:
                    return json_obj
                
                print(f"Directly parse JSON failed")
                print(f"Original text: {text[:200]}..." if len(text) > 200 else text)
                
                try:
                    if '"direct_obj"' in text and '"potential_obj"' in text:
                        result = {"direct_obj": [], "potential_obj": []}
                        
                        direct_match = re.search(r'"direct_obj"\s*:\s*(\[.*?\])', text, re.DOTALL)
                        if direct_match:
                            direct_str = direct_match.group(1)
                            direct_str = re.sub(r',\s*]', ']', direct_str)
                            try:
                                direct_list = json.loads(direct_str)
                                result["direct_obj"] = direct_list
                            except:
                                pass
                        
                        potential_match = re.search(r'"potential_obj"\s*:\s*(\[.*?\])', text, re.DOTALL)
                        if potential_match:
                            potential_str = potential_match.group(1)
                            potential_str = re.sub(r',\s*]', ']', potential_str)
                            try:
                                potential_list = json.loads(potential_str)
                                result["potential_obj"] = potential_list
                            except:
                                pass
                        
                        instr_match = re.search(r'"instruction"\s*:\s*"(.*?)"', text)
                        if instr_match:
                            result["instruction"] = instr_match.group(1)
                        
                        if result["direct_obj"] or result["potential_obj"]:
                            return result
                except Exception as e:
                    print(f"Failed to extract JSON using regex: {e}")
                
                print(f"Warning: Failed to extract JSON from text, returning empty object.")
                return {"direct_obj": [], "potential_obj": []}
            
            pred_json = extract_json_from_text(pred)
            label_json = extract_json_from_text(label)
            
            pred_direct_obj = set(pred_json.get('direct_obj', []))
            label_direct_obj = set(label_json.get('direct_obj', []))
            
            pred_potential_obj = set(pred_json.get('potential_obj', []))
            label_potential_obj = set(label_json.get('potential_obj', []))
            
            direct_precision = len(pred_direct_obj.intersection(label_direct_obj)) / max(len(pred_direct_obj), 1)
            direct_recall = len(pred_direct_obj.intersection(label_direct_obj)) / max(len(label_direct_obj), 1)
            direct_f1 = 2 * (direct_precision * direct_recall) / max((direct_precision + direct_recall), 1e-6)
            direct_f1_scores.append(direct_f1)
            
            potential_precision = len(pred_potential_obj.intersection(label_potential_obj)) / max(len(pred_potential_obj), 1)
            potential_recall = len(pred_potential_obj.intersection(label_potential_obj)) / max(len(label_potential_obj), 1)
            potential_f1 = 2 * (potential_precision * potential_recall) / max((potential_precision + potential_recall), 1e-6)
            potential_f1_scores.append(potential_f1)
            
            pred_all_obj = pred_direct_obj.union(pred_potential_obj)
            label_all_obj = label_direct_obj.union(label_potential_obj)
            
            total_precision = len(pred_all_obj.intersection(label_all_obj)) / max(len(pred_all_obj), 1)
            total_recall = len(pred_all_obj.intersection(label_all_obj)) / max(len(label_all_obj), 1)
            total_f1 = 2 * (total_precision * total_recall) / max((total_precision + total_recall), 1e-6)
            total_f1_scores.append(total_f1)
            
            relevance_scores = {}
            
            # Direct objects in labels have the highest relevance
            for obj in label_direct_obj:
                relevance_scores[obj] = 2
            
            # Potential objects in labels have the second highest relevance
            for obj in label_potential_obj:
                if obj not in relevance_scores:  # Avoid overwriting direct objects
                    relevance_scores[obj] = 1
            
            # Calculate DCG
            dcg = 0.0
            # First consider predicted direct objects
            for i, obj in enumerate(pred_direct_obj):
                if obj in relevance_scores:
                    # DCG formula: rel_i / log2(i+2)
                    dcg += relevance_scores[obj] / np.log2(i + 2)
            
            # Then consider predicted potential objects (after direct objects)
            for i, obj in enumerate(pred_potential_obj):
                if obj in relevance_scores and obj not in pred_direct_obj:  # Avoid double counting
                    # DCG formula
                    dcg += relevance_scores[obj] / np.log2(i + len(pred_direct_obj) + 2)
            
            # Calculate IDCG (ideal case)
            # Sort all relevant objects by relevance
            sorted_relevance = sorted([(obj, score) for obj, score in relevance_scores.items()],
                                     key=lambda x: x[1], reverse=True)
            
            idcg = 0.0
            for i, (obj, score) in enumerate(sorted_relevance):
                idcg += score / np.log2(i + 2)
            
            # Calculate NDCG
            ndcg = dcg / max(idcg, 1e-6)
            ndcg_scores.append(ndcg)
            
            # Calculate weighted score for single sample
            weighted_score = 0.4 * direct_f1 + 0.3 * potential_f1 + 0.3 * ndcg
            weighted_scores.append(weighted_score)
            
            # Try to get instruction text
            instruction = ""
            if isinstance(label_json, dict) and "instruction" in label_json:
                instruction = label_json.get("instruction", "")
            elif isinstance(pred_json, dict) and "instruction" in pred_json:
                instruction = pred_json.get("instruction", "")
            
            # Save detailed metrics for each sample
            sample_metric = {
                'index': idx,
                'instruction': instruction,
                'direct_obj_f1': direct_f1,
                'potential_obj_f1': potential_f1,
                'total_f1': total_f1,
                'ndcg': ndcg,
                'weighted_score': weighted_score,
                'pred_direct_obj': list(pred_direct_obj),
                'label_direct_obj': list(label_direct_obj),
                'pred_potential_obj': list(pred_potential_obj),
                'label_potential_obj': list(label_potential_obj)
            }
            per_sample_metrics.append(sample_metric)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            print(f"Prediction: {pred[:200]}..." if len(str(pred)) > 200 else pred)
            print(f"Label: {label[:200]}..." if len(str(label)) > 200 else label)
            # Error samples score 0
            direct_f1_scores.append(0.0)
            potential_f1_scores.append(0.0)
            total_f1_scores.append(0.0)
            ndcg_scores.append(0.0)
            weighted_scores.append(0.0)
            
            # Add metrics for error samples
            sample_metric = {
                'index': idx,
                'instruction': "",
                'direct_obj_f1': 0.0,
                'potential_obj_f1': 0.0,
                'total_f1': 0.0,
                'ndcg': 0.0,
                'weighted_score': 0.0,
                'error': str(e)
            }
            per_sample_metrics.append(sample_metric)
    
    # Calculate average metrics
    metrics['direct_obj_f1'] = sum(direct_f1_scores) / total_samples
    metrics['potential_obj_f1'] = sum(potential_f1_scores) / total_samples
    metrics['total_f1'] = sum(total_f1_scores) / total_samples
    metrics['ndcg'] = sum(ndcg_scores) / total_samples
    metrics['weighted_score'] = sum(weighted_scores) / total_samples
    
    return metrics, per_sample_metrics


prompt_template = """
## Objective
- Add two new fields to each JSON entry: `direct_obj` and `potential_obj`, based on the `instruction` field.
  - **`direct_obj`**: A list of all objects explicitly mentioned in `instruction` (e.g., `["sink", "table"]` in "clean the sink and table").
  - **`potential_obj`**: A list of objects reasonably inferred based on the task or room type, describing the environment or related to the task.
- Retain original fields ( instruction`) unchanged in the output.

## Processing Steps

### Identify Direct Object (`direct_obj`)
- Extract all objects **explicitly named** in `instruction` as targets of the main task or verbs.
  - Focus on nouns syntactically tied to task-related verbs (e.g., "sink" and "table" in "clean the sink and table").
  - Use syntactic parsing (e.g., dependency parsing) to identify direct objects of verbs when possible.
  - Include all objects explicitly mentioned as task targets, even if tied to different verbs (e.g., "clean the sink and organize the table" → `["sink", "table"]`).
- Sort `direct_obj` using:
  1. **Frequency (40%)**: Objects mentioned multiple times rank higher.
  2. **Task Relevance (40%)**: Objects tied to the primary task or verb rank higher (e.g., "sink" in "clean the sink and check the table").
  3. **Order of Mention (20%)**: Earlier-mentioned objects rank higher if frequency and relevance are equal.
- If no objects are mentioned (e.g., "go to the spa") or only pronouns/vague terms are used (e.g., "clean it"), set `direct_obj` to `[]`.
- Match nouns exactly as in `instruction` (e.g., "sink", not "basin").
- Do not infer objects for `direct_obj`; they must be explicitly stated.

### Identify Potential Objects (`potential_obj`)
- Include objects that are:
  - **Inferred** based on the task or room type, up to a maximum of 3 inferred objects (5 for vague instructions), describing the environment or context (e.g., "bed", "tiles" in "Go to the spa with one bed, brown tiles").
  - Guidelines for inference:
  - Select inferred objects most relevant to the task (e.g., "sponge" for cleaning) or room type (e.g., "towel" in a spa).
  - Avoid speculative inferences (e.g., do not infer "chandelier" in a spa unless mentioned).
  - Use singular nouns for inferred objects unless context suggests plural.
  - If more than 3 (or 5 for vague instructions) inferred objects are possible, prioritize by typicality (e.g., "sponge" over "toaster" for cleaning in a kitchen).
- Sort `potential_obj` by:
  1. **Explicit Mention (40%)**: Explicitly mentioned objects rank higher.
  2. **Frequency (30%)**: Objects mentioned multiple times rank higher.
  3. **Task Relevance (30%)**: Objects closer to the task or central to the environment rank higher (e.g., "sponge" for cleaning over "lamp").
- If no objects are inferred, set `potential_obj` to `[]`.

### Handle Special Cases
- **Vague Instructions**:
  - An instruction is vague if it lacks specific object nouns (e.g., "clean the room") or uses generic verbs without clear targets (e.g., "fix something").
  - Set `direct_obj` to `[]` and infer up to 5 typical objects for `potential_obj` based on room type (e.g., ["table", "chair", "lamp"] for a generic room).
- **Compound Objects**:
  - Include all explicitly mentioned task targets in `direct_obj` (e.g., "clean the sink and table" → `["sink", "table"]`), sorted as above.
- **Non-Physical Objects**:
  - Exclude abstract entities (e.g., "mess" in "clean the mess") from both `direct_obj` and `potential_obj`, setting to `[]`.
  - Allow inferred physical objects relevant to the task (e.g., "sponge" for cleaning).
- **Multi-Room Instructions**:
  - Infer `potential_obj` based on the room where the task occurs (e.g., spa for "go from kitchen to spa and clean the sink"). If unclear, use the last-mentioned room.
- **Empty/Malformed Instructions**:
  - If `instruction` is empty, null, or malformed (e.g., ""), set `direct_obj` to `[]` and `potential_obj` to `[]`.
  
### Input Validation
- Validate input JSON before processing:
  - If `instruction` is missing, set `direct_obj` to `[]` and `potential_obj` to `[]`.
  - If other required fields (`path`, `heading`, `scan`, `path_id`, `instr_id`) are missing, retain them as null or their default type (e.g., empty list for `path`) and proceed.

## Output Format
- Generate a JSON dictionary with fields in order: `instruction`, `direct_obj`, `potential_obj`.
- Ensure:
  - `direct_obj` is a list of strings, sorted by frequency, task relevance, order of mention, and alphabetical tiebreaker.
  - `potential_obj` is a list of strings, sorted by explicit mention, frequency, task relevance, and alphabetical tiebreaker.
  - JSON is well-formed, with 2-space indentation, no trailing commas, and consistent double quotes.
  - Original fields (`instruction`) are unchanged.
- Validate output:
  - All required fields are present in the specified order.
  - `direct_obj` and `potential_obj` are lists of strings.
  - JSON syntax is correct with no trailing commas or missing brackets.

## Notes
- **Exact Matching for `direct_obj`**: Match nouns exactly as in `instruction`.
- **Inference for `potential_obj`**: Limited to 5 inferred objects to ensure relevance.
- **Language Consistency**: Use singular/plural as in `instruction` for mentioned objects; inferred objects use singular unless context suggests plural.

## Examples

### Example 1
**Input**:
```json
{
    "instruction": "Go to the spa with one bed, brown tiles on the walls, a visible white radiator, and clean out the sink and bed"
}
```

**Output**:
```json
{
  "instruction": "Go to the spa with one bed, brown tiles on the walls, a visible white radiator, and clean out the sink and bed",
  "direct_obj": ["sink", "bed", "tiles", "radiator", ],
  "potential_obj": ["towel", "tub"]
}
```

## Task
- Process the JSON input provided by me and return a complete JSON output adhering to the above requirements.
- Ensure the output is correctly formatted, readable, and both `direct_obj` and `potential_obj` are sorted by specified criteria.
- Do not output any other text or comments. /no_think/ /nothink/
- Wait for the my JSON input to process. Do not process sample inputs unless explicitly provided.
"""

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Evaluate object recognition model performance on VLN tasks')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B', help='Model name')
    parser.add_argument('--checkpoint', type=str, required=False, help='Checkpoint path', default="xxx/checkpoint")
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--dataset', type=str, default='xxx/xxx.json', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--start', type=int, default=0, help='Limit the number of samples to evaluate')
    parser.add_argument('--end', type=int, default=3000, help='Limit the number of samples to evaluate')
    parser.add_argument('--output', type=str, default="xxx/xxx.json", help='Result output path, if not specified, only print')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU model loading')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5,6,7", help='Specify the GPU ID for model loading, e.g., "0,1,2,3", only effective when multi_gpu=True')
    
    args = parser.parse_args()
    
    # Set CUDA environment variable
    if args.multi_gpu and args.gpu_ids:
        print(f"Enable multi-GPU mode, using GPU: {args.gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    else:
        print(f"Single-GPU mode, using GPU: {args.device}")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    # Set memory optimization environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Model configuration
    model_name = args.model
    # lora_checkpoint = safe_snapshot_download(args.checkpoint)
    print(f"Model: {model_name}")
    # print(f"Checkpoint: {lora_checkpoint}")
    
    # System prompt
    # default_system = "\nYou are a Vision-and-Language Navigation (VLN) agent designed to interpret navigation and task instructions in household environments.\n"
    template_type = None
    
    # Load model and dialog template
    print("Loading model...")
    
    # Check if it's a large model
    is_large_model = "32B" in model_name
    
    # If it's a large model and multi-GPU is enabled, automatically adjust parameters
    if is_large_model:
        print(f"Detected large model ({model_name})")
        if args.multi_gpu:
            print("Using multi-GPU distributed model loading")
            # Large models recommend using a smaller batch_size
            if args.batch_size > 32:
                original_batch_size = args.batch_size
                args.batch_size = 32
                print(f"Large models recommend using a smaller batch size, which has been adjusted from {original_batch_size} to {args.batch_size}")
        else:
            print("Warning: Loading large model but multi-GPU mode is not enabled, which may lead to memory shortage")
    
    # Clean CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Configure multi-GPU loading parameters
    model_kwargs = {}
    if args.multi_gpu:
        # Automatically distribute the model to all available GPUs
        model_kwargs["device_map"] = "auto"
        # For Flash Attention 2 support, depends on your environment
        if torch.cuda.get_device_capability()[0] >= 8:  # For Ampere and above architectures
            print("Enable Flash Attention 2 to improve performance and reduce memory usage")
            model_kwargs["attn_implementation"] = "flash_attention_3"
    else:
        # Single-GPU mode
        model_kwargs["device_map"] = f"cuda:{args.device}"
    
    try:
        model, tokenizer = get_model_tokenizer(model_name, **model_kwargs)
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    # Load Swift adapter
    swift_kwargs = {}
    if args.multi_gpu:
        swift_kwargs["device_map"] = "auto"
        swift_kwargs["max_memory"] = {0: f"10GiB", "cpu": "60GiB"}
    engine = PtEngine(model_name, max_batch_size=2)
    request_config = RequestConfig(max_tokens=2048, temperature=0)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        n = len(dataset)
        end = min(n, args.end)
        dataset = dataset[args.start:n]
    
    print(f"Evaluate sample number: {len(dataset)}")
    
    # Prepare inference requests
    infer_requests = []
    labels = []
    for item in dataset:
        input_json = {
            "instruction": item["instruction"]
        }
        
        # Build label format
        label_json = {
            "instruction": item["instruction"],
            "direct_obj": item.get("direct_obj", []),
            "potential_obj": item.get("potential_obj", [])
        }
        
        # Build complete prompt
        user_msg = prompt_template + json.dumps(input_json, indent=2)
        
        # Add to request list
        infer_requests.append(InferRequest(messages=[{'role': 'user', 'content': user_msg}]))
        labels.append(json.dumps(label_json, indent=2))
    
    # Batch inference
    all_preds = []
    batch_size = args.batch_size
    
    print(f"Batch size: {batch_size}")
    for i in range(0, len(infer_requests), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(infer_requests) + batch_size - 1)//batch_size}")
        batch_requests = infer_requests[i:i+batch_size]
        
        # Clean CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        resp_list = engine.infer(batch_requests, request_config)
        for resp in resp_list:
            all_preds.append(resp.choices[0].message.content)
    
    # Calculate and display metrics
    print("Calculating evaluation metrics...")
    metrics, per_sample_metrics = compute_json_metrics(
        None,
        tokenizer=tokenizer,
        decoded_preds_and_labels=(all_preds, labels)
    )
    
    # Output results
    result_str = json.dumps(metrics, indent=2)
    print("\n=== Evaluation results ===")
    print(result_str)
    
    # Save results
    if args.output:
        checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint))
        output_file = args.output
        # If the output is a directory, create a file in the directory
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, f"results_{checkpoint_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'checkpoint': args.checkpoint,
                'device': args.device if not args.multi_gpu else args.gpu_ids,
                'multi_gpu': args.multi_gpu,
                'dataset': args.dataset,
                'num_samples': len(dataset),
                'metrics': metrics
            }, f, indent=2)
        print(f"Results saved to: {output_file}")
        
        # Save detailed metrics to a separate file
        detailed_output_file = output_file.replace('.json', '_detailed.json')
        with open(detailed_output_file, 'w', encoding='utf-8') as f:
            json.dump(per_sample_metrics, f, indent=2, ensure_ascii=False)
        print(f"Detailed metrics have been saved to: {detailed_output_file}")
        
        with open("xxx/infer/qwen4b_REVERIE_cal_seen_obj_results_ori_all.json", "w", encoding="utf-8") as f:
            json.dump(all_preds, f, indent=2)

if __name__ == "__main__":
    main()