import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.generation.utils import GenerationConfig



class Predictor:
    def __init__(self, input_file_list, model_name, save_dir, device):
        self.model_name = model_name
        # self.model = model
        # self.tokenizer = tokenizer
        # self.processor = processor
        self.device = device
        self.input_data_list = input_file_list
        self.save_dir = save_dir

    
    def load_model_and_tokenizer(self,): pass

    def generate_prediction(self, model, tokenizer, processor, messages): pass

    def build_input_prompt(self,): pass

    def load_data(self):
        for filename in self.input_data_list:
            with open(filename, 'r') as f:
                prompts = json.load(f)
            yield prompts

    def predict(self,):
        model, tokenizer, processor = self.load_model_and_tokenizer(self.model_name, self.device)
        predictions = []
        idx = 0
        for input_data in self.load_data():
            messages = self.build_input_prompt(input_data)
            result = self.generate_prediction(model, tokenizer, processor, messages, self.device)
            predictions.append(result)
            basename = os.path.basename(self.input_data_list[idx])
            savename = basename.replace('.json', '.txt').replace('_prompts', '')
            self.save_prediction(result, f'{self.save_dir}/{savename}')
            idx += 1
        
        return predictions
    
    def save_prediction(self, prediction, save_path):
        with open(save_path, 'w') as f:
            f.write(prediction)


class Qwen2VLPredictor(Predictor):
    def load_model_and_tokenizer(path, device):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model = model.eval()
        return model, tokenizer, processor
    
    def build_input_prompt(self, input_data):
        image_link = input_data['image_link']
        messages = []
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
                            "image": image_link
                        },
                        {
                            "type": "text",
                            "text": input_data['suffix'] + '\n\n' + f"Read the document and answer: {input_data['retrieval_question']}"
                        }
                    ]
                },
            ]
        return messages
    
    def generate_prediction(self, model, tokenizer, processor, messages):
        from qwen_vl_utils import process_vision_info
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # print(image_inputs)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(pred)
        return pred.strip()




if __name__ == '__main__':
    with open('configs/config-pred.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor = Qwen2VLPredictor(
        input_file_list=glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'),
        model_name=model_name,
        save_dir=save_dir,
        device=device
    )
    predictions = predictor.predict()


