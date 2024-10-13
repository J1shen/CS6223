import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers.generation.utils import GenerationConfig



def build_input_prompt(input_data, model_name):
    image_link = input_data['image_link']
    messages = []
    if "qwen2-vl" in model_name:
        if image_link is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data['prefix'] + ' ' + input_data['needle'] + ' ' + input_data['suffix'] + \
                                        '\n\n' + f"Read the document and answer: {input_data['retrieval_question']}"
                        }
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data['prefix']
                        },
                        {
                            "type": "image",
                            "content": image_link
                        },
                        {
                            "type": "text",
                            "text": input_data['suffix'] + '\n\n' + f"Read the document and answer: {input_data['retrieval_question']}"
                        }
                    ]
                },
            ]
    return messages



def pred(model_name, model, tokenizer, processor, input_data, device, max_new_tokens=1024, temperature=0.1):
    model_name = model_name.lower()
    messages = build_input_prompt(input_data, model_name)
    if "qwen2-vl" in model_name:
        from qwen_vl_utils import process_vision_info
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return pred.strip()

def load_model_and_tokenizer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    return model, tokenizer, processor

if __name__ == '__main__':
    with open('config-pred.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer, processor = load_model_and_tokenizer(model_name, device)

    for filename in glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'):
        with open(filename, 'r') as f:
            prompts = json.load(f)

        result = pred(model_name.lower(), model, tokenizer, processor, prompts, device)

        basename = os.path.basename(filename)
        newname = basename.replace('.json', '.txt').replace('_prompts', '')
        with open(f'{save_dir}/{newname}', 'w') as f:
            f.write(result)

