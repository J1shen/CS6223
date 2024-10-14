from transformers import AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
from predictors import Predictor
from PIL import Image
import requests

class LlamaVLPredictor(Predictor):
    def __init__(self, model_name, input_data_list, save_dir, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = MllamaForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model = model.eval()
        
        super().__init__(model_name, model, tokenizer, processor, input_data_list, save_dir, device)
    
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
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": input_data['suffix'] + '\n\n' + f"Read the document and answer: {input_data['retrieval_question']}"
                        }
                    ]
                },
            ]
        return messages
    
    def generate_prediction(self, messages, image_link=None):
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        image_inputs = Image.open(requests.get(image_link, stream=True).raw) if image_link else None
        
        # print(image_inputs)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        pred = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return pred.strip()




if __name__ == '__main__':
    import os
    import yaml
    import glob
    import torch
    with open('configs/config-pred-llamavl.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor = LlamaVLPredictor(
        input_data_list=glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'),
        model_name=model_name,
        save_dir=save_dir,
        device=device
    )
    predictions = predictor.predict()