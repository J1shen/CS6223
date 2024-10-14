import os
import json

class Predictor:
    def __init__(
        self, 
        model_name,
        model,
        tokenizer,
        processor, 
        input_data_list, 
        save_dir, 
        device
        ):
        
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.input_data_list = input_data_list
        self.save_dir = save_dir

    
    def load_model_and_tokenizer(self,): 
        pass

    def generate_prediction(self,): 
        pass

    def build_input_prompt(self,): 
        pass

    def load_data(self):
        for filename in self.input_data_list:
            with open(filename, 'r') as f:
                prompts = json.load(f)
            yield prompts

    def predict(self,):
        predictions = []
        idx = 0
        for input_data in self.load_data():
            messages = self.build_input_prompt(input_data)
            result = self.generate_prediction(messages, image_link=input_data['image_link'])
            predictions.append(result)
            basename = os.path.basename(self.input_data_list[idx])
            savename = basename.replace('.json', '.txt').replace('_prompts', '')
            self.save_prediction(result, f'{self.save_dir}/{savename}')
            idx += 1
        
        return predictions
    
    def save_prediction(self, prediction, save_path):
        with open(save_path, 'w') as f:
            f.write(prediction)