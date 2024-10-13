import random
import numpy as np
from datasets import load_dataset
from generators.generator import Generator

class generator_sample(Generator):
    def __init__(self, text_datasets: list[str], img_datasets: list[str]) -> None:
        self.text_datasets = []
        self.img_datasets = []
        
        self.text_datasets_len = []
        self.img_datasets_len = []
        for text_dataset in text_datasets:
            self.text_datasets.append(
                load_dataset(text_dataset, split="train")
            )
            self.text_datasets_len.append(
                len(self.text_datasets[-1])
            )
            
        for img_dataset in img_datasets:
            self.img_datasets.append(
                load_dataset(img_dataset, split="train")
            )
            self.img_datasets_len.append(
                len(self.img_datasets[-1])
            )
            
        super().__init__()
            
    def generate(self):
        textset_idx = random.randint(0, len(self.text_datasets)-1)
        imgset_idx = random.randint(0, len(self.img_datasets)-1)
        
        text_idx = random.randint(0, self.text_datasets_len[textset_idx] - 1)
        img_idx = random.randint(0, self.img_datasets_len[imgset_idx] - 1)
        
        text_input = self.text_datasets[textset_idx][text_idx]['instruction']
        img_input = self.img_datasets[imgset_idx][img_idx]
        
        question_position = random.randint(0, 3)
        if question_position == 0:
            answer = img_input['human_caption'][0]
            question = self.mask_text(answer)
        else:
            sentences = text_input.split('. ')
            num_sentences = len(sentences)
            lambda_ = 1.5
            if question_position == 1:
                probabilities = np.exp(-np.arange(num_sentences) / lambda_)
                probabilities /= np.sum(probabilities)
                index = np.random.choice(num_sentences, p=probabilities)
            elif question_position == 2:
                probabilities = np.exp(np.arange(num_sentences) / lambda_) 
                probabilities /= np.sum(probabilities)               
                index = np.random.choice(num_sentences, p=probabilities)
            elif question_position == 3:
                indices = np.arange(num_sentences)
                mean_index = np.mean(indices)
                std_dev = 1
                probabilities = np.exp(-0.5 * ((indices - mean_index) / std_dev) ** 2)
                probabilities /= np.sum(probabilities)
                index = np.random.choice(num_sentences, p=probabilities)
            
            answer = sentences[index]
            question = self.mask_text(answer)
            
        return {
            'text_input': text_input,
            'img_input': img_input['image'],
            'question': question,
            'answer': answer
        }
    
if __name__ == '__main__':
    text_datasets = ['Yukang/LongAlpaca-12k']
    img_datasets = ['OpenFace-CQUPT/HumanCaption-10M']
    generator = generator_sample(text_datasets, img_datasets)
    case = generator.generate()
    print(case)
