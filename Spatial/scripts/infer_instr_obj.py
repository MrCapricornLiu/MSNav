import json
import re
import os
from tqdm import tqdm
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

# Environment settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Prompt template
prompt_template = """
You are a Vision-and-Language Navigation (VLN) agent designed to interpret navigation and task instructions in household environments.
"""
user_content = """
Analyze the given instruction to infer `direct_obj` (objects mentioned in the instruction, sorted by importance) and `potential_obj` (other relevant objects, sorted by relevance). Return a JSON object with `direct_obj` and `potential_obj` lists.
"""

# Model configuration
model = 'Qwen/Qwen3-4B'
lora_checkpoint = safe_snapshot_download('xxx/checkpoint')
template_type = None  # Use the default template_type for the corresponding model
default_system = prompt_template  # Use the specified system prompt

# Load model and dialog template
model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=8)  # Increase batch size
request_config = RequestConfig(max_tokens=512, temperature=0)

def clean_llm_json_output(raw_string):
    """Clean and parse the JSON string output from the LLM"""
    if not isinstance(raw_string, str):
        print("Error: Input must be a string.")
        return None

    # Find code blocks formatted as ```json ... ```
    json_code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_string)
    
    # If JSON code blocks are found
    if json_code_blocks:
        for block in json_code_blocks:
            # Fix trailing comma issues
            fixed_block = re.sub(r',(\s*)]', r'\1]', block)
            fixed_block = re.sub(r',(\s*)}', r'\1}', fixed_block)
            # Try to parse the repaired block
            try:
                return json.loads(fixed_block)
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects directly
    try:
        match = re.search(r'(\{[\s\S]*\})', raw_string)
        if match:
            potential_json = match.group(1)
            fixed_json = re.sub(r',(\s*)]', r'\1]', potential_json)
            fixed_json = re.sub(r',(\s*)}', r'\1}', fixed_json)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try the original cleaning method
    content = raw_string.strip()
    
    # Remove possible Markdown code block markers
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-len("```")].strip()

    # Remove JavaScript-style comments
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        in_quotes = False
        comment_start_index = -1
        for i, char in enumerate(line):
            if char == '"':
                if i == 0 or line[i-1] != '\\':
                    in_quotes = not in_quotes
            elif char == '/' and i > 0 and line[i-1] == '/' and not in_quotes:
                comment_start_index = i - 1
                break
        
        if comment_start_index != -1:
            cleaned_line = line[:comment_start_index].rstrip()
        else:
            cleaned_line = line

        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)

    cleaned_json_string = "\n".join(cleaned_lines)
    
    # Fix trailing comma issues
    cleaned_json_string = re.sub(r',(\s*)]', r'\1]', cleaned_json_string)
    cleaned_json_string = re.sub(r',(\s*)}', r'\1}', cleaned_json_string)

    # Parse the cleaned string
    try:
        parsed_data = json.loads(cleaned_json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        try:
            start_idx = raw_string.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(raw_string)):
                    if raw_string[i] == '{':
                        brace_count += 1
                    elif raw_string[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_part = raw_string[start_idx:i+1]
                            fixed_json = re.sub(r',(\s*)]', r'\1]', json_part)
                            fixed_json = re.sub(r',(\s*)}', r'\1}', fixed_json)
                            try:
                                return json.loads(fixed_json)
                            except json.JSONDecodeError:
                                pass
        except Exception:
            pass
            
        print(f"All JSON parsing methods failed: {e}")
        
        # Try manual repair of trailing comma issues
        try:
            fixed_json = re.sub(r',(\s*)]', r'\1]', cleaned_json_string)
            fixed_json = re.sub(r',(\s*)}', r'\1}', fixed_json)
            parsed_data = json.loads(fixed_json)
            print("Successfully parsed JSON by removing trailing commas")
            return parsed_data
        except Exception:
            pass
        
        # Create an empty dictionary to preserve the original instruction
        for line in raw_string.splitlines():
            if '"instruction"' in line:
                instruction_match = re.search(r'"instruction"\s*:\s*"([^"]*)"', line)
                if instruction_match:
                    instruction = instruction_match.group(1)
                    return {
                        "instruction": instruction,
                        "direct_obj": [],
                        "potential_obj": []
                    }
        
        return {"direct_obj": [], "potential_obj": []} 

def process_batch(batch_data, batch_size=8):
    """Process in batch"""
    batch_messages = []
    for item in batch_data:
        instruction = item.get("instructions", [])[0]
        message = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"{user_content}\nInstruction: {instruction}"}
        ]
        batch_messages.append(InferRequest(messages=message))
    
    resp_list = engine.infer(batch_messages, request_config)
    
    for i, (item, resp) in enumerate(zip(batch_data, resp_list)):
        response = resp.choices[0].message.content
        parsed_json = clean_llm_json_output(response)
        
        if parsed_json:
            item["direct_obj"] = parsed_json.get("direct_obj", [])
            item["potential_obj"] = parsed_json.get("potential_obj", [])
        else:
            item["direct_obj"] = []
            item["potential_obj"] = []
    
    return batch_data

def main():
    input_file = "xxx.json"
    output_file = "xxx.json"
    
    batch_size = 8
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"length: {len(data)}")
    
    processed_data = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        processed_batch = process_batch(batch, batch_size)
        processed_data.extend(processed_batch)

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"Process finish {i+len(batch)} samples, saving results...")
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Saved in {output_file}")

if __name__ == "__main__":
    main()