import random
import torch
import numpy as np
from datasets import load_dataset
from generators.generator import Generator
from diffusers import (
    FluxPipeline,
    DiffusionPipeline
)

class generator_diffuse(Generator):
    def __init__(
        self, 
        text_datasets: list[str],
        image_model: str
        ) -> None:
        self.text_datasets = []
        
        self.text_datasets_len = []
        for text_dataset in text_datasets:
            self.text_datasets.append(
                load_dataset(text_dataset, split="train")
            )
            self.text_datasets_len.append(
                len(self.text_datasets[-1])
            )
        
        self.image_model = self.load_image_model(image_model)
        
        super().__init__()
    
    def load_image_model(self, image_model: str, device='cuda'):
        if 'Flux' in image_model:
            pipe = FluxPipeline.from_pretrained(
                image_model, 
                torch_dtype=torch.bfloat16
                )
            pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        elif 'diffusion' in image_model:
            pipe = DiffusionPipeline.from_pretrained(
                image_model,
                torch_dtype=torch.float16
                ).to(device)
        else:
            raise ValueError("Image model not supported")
        return pipe
    
    def generate(self):
        textset_idx = random.randint(0, len(self.text_datasets)-1)
        
        text_idx = random.randint(0, self.text_datasets_len[textset_idx] - 1)
        
        text_input = self.text_datasets[textset_idx][text_idx]['instruction']
        
        image_sentence = random.choice(text_input.split('. '))
        img_input = self.image_model(
                        prompt=image_sentence,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        max_sequence_length=512,
                        generator=torch.Generator("cpu").manual_seed(0),
                    ).images[0]
        
        question_position = random.randint(0, 3)
        if question_position == 0:
            answer = image_sentence
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
            'img_input': img_input,
            'question': question,
            'answer': answer
        }
    
if __name__ == '__main__':
    text_datasets = ['Yukang/LongAlpaca-12k']
    generator = generator_diffuse(
        text_datasets, 
        "black-forest-labs/FLUX.1-dev"
        )
    case = generator.generate()
    print(case)
