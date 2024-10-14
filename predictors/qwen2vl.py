from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from predictor import Predictor

class Qwen2VLPredictor(Predictor):
    def __init__(self, model_name, input_data_list, save_dir, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
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
    
    def generate_prediction(self, messages, image_link=None):
        from qwen_vl_utils import process_vision_info
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # print(image_inputs)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(pred)
        return pred.strip()




if __name__ == '__main__':
    import os
    import yaml
    import glob
    import torch
    with open('configs/config-pred-qwen2vl.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor = Qwen2VLPredictor(
        input_data_list=glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'),
        model_name=model_name,
        save_dir=save_dir,
        device=device
    )
    predictions = predictor.predict()