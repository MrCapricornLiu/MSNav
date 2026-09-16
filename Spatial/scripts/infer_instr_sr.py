import json
import re
import os
from tqdm import tqdm

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

prompt_template = """
You are a Vision-and-Language Navigation (VLN) agent designed to analyze the given instruction to infer spatial layout of the final destination. Return two sentences:"The destination is ..." describing the final location. "The spatial layout of the destination is ..." describing the arrangement of key objects based on the instruction.
"""

model = 'Qwen/Qwen3-4B'
lora_checkpoint = safe_snapshot_download('xxx/checkpoint') # Using your checkpoint
template_type = None  
default_system = prompt_template  

model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=8)  # Increase batch size
request_config = RequestConfig(max_tokens=512, temperature=0)

def extract_spatial_relations(text):
    """
    Extract spatial relations from text, assuming the text contains two sentences:
    1. "The destination is ..."
    2. "The spatial layout of the destination is ..."
    
    Return a list of these two sentences.
    """
    if not isinstance(text, str):
        return []
    
    text = text.strip()
    
    dest_match = re.search(r"The destination is[^\.]+\.", text, re.IGNORECASE)
    layout_match = re.search(r"The spatial layout of the destination is[^\.]+\.", text, re.IGNORECASE)
    
    if dest_match and layout_match:
        return [dest_match.group(0), layout_match.group(0)]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    
    for sentence in sentences:
        if "destination is" in sentence.lower() and len(result) < 2:
            result.append(sentence.strip())
    
    if len(result) < 2:
        sentences = text.split('.')
        valid_sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return valid_sentences[:2] if len(valid_sentences) >= 2 else valid_sentences
    
    return result

def format_destination_text(text_list):
    """
    Format the two sentences into the target format.
    """
    if not text_list or len(text_list) == 0:
        return []
    
    if len(text_list) == 1:
        return text_list
    
    first_part = text_list[0]
    second_part = text_list[1]
    
    if second_part:
        second_part = second_part[0].lower() + second_part[1:]
    
    return [first_part, second_part]

def process_batch(batch_data, batch_size=8):
    """Process in batch"""
    batch_messages = []
    for item in batch_data:
        instruction = item.get("instructions", [])[0]
        message = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"Instruction: {instruction}\nWhat is the final destination and its spatial layout of this instruction?"}
        ]
        batch_messages.append(InferRequest(messages=message))
    
    resp_list = engine.infer(batch_messages, request_config)
    
    for i, (item, resp) in enumerate(zip(batch_data, resp_list)):
        response = resp.choices[0].message.content
        
        spatial_relations = extract_spatial_relations(response)
        
        formatted_relations = format_destination_text(spatial_relations)
        
        item["final_destination_spatial_relations"] = formatted_relations[0] + ' ' + formatted_relations[1]
        # print(item)
        # input()
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