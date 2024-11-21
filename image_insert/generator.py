from transformers import pipeline
from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
import json

class Generator:
    def __init__(
        self,
        language_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        image_model: str = 'stabilityai/stable-diffusion-3.5-large',
        ):
        self.llm = pipeline('text-generation', model=language_model, torch_dtype=torch.bfloat16, device='cuda')
        self.image_model = StableDiffusion3Pipeline.from_pretrained(image_model, torch_dtype=torch.bfloat16).to('cuda')
        
        self.img_message = [
                {"role": "system", "content": "You are a chatbot who generates sentences."},
                {"role": "user", "content": 'Give a sentence to discribe an simple image of something before white background.'},
            ]
        self.text_message = [
                {"role": "system", "content": "You are a chatbot who generates sentences."},
                {"role": "user", "content": 'Give a sentence within 10 words including a fact.'},
            ]
    
    def insert(self, image: Image.Image, text: str):
        draw = ImageDraw.Draw(image)
        
        font = ImageFont.truetype("image_insert/Arial.ttf", 40)

        img_width, img_height = image.size

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        rotation_angle = random.randint(0, 360)
        text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
        ImageDraw.Draw(text_img).text((0, 0), text, font=font, fill=(255, 0, 0, 255))
        rotated_text_img = text_img.rotate(rotation_angle, expand=1)
        rotated_width, rotated_height = rotated_text_img.size

        max_x = max(0, img_width - rotated_width)
        max_y = max(0, img_height - rotated_height)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        image.paste(rotated_text_img, (x, y), rotated_text_img)
        return image, rotation_angle
    
    def generate(self, num: int = 10, output_path: str = 'output'):
        results = {}
        img_inputs = self.llm(self.img_message, max_new_tokens=32, num_return_sequences=num, temperature=1.2)
        img_inputs = [img_input["generated_text"][-1]['content'] for img_input in img_inputs]
        
        text_inputs = self.llm(self.text_message, max_new_tokens=16, num_return_sequences=num, temperature=1.2)
        text_inputs = [text_input["generated_text"][-1]['content'] for text_input in text_inputs]
        
        for i, (img_input, text_input) in tqdm(enumerate(zip(img_inputs, text_inputs))):
            
            img_output = self.image_model(
                prompt=img_input,
                num_inference_steps=28,
                guidance_scale=3.5,
                ).images[0]
            
            mutated_img, angle = self.insert(img_output, text_input)
            path = f'{output_path}/{i}.png'
            mutated_img.save(path)
            
            results[i] = {
                'image': path,
                'text': text_input,
                'angle': angle,
            }
        
        json.dump(results, open(f'{output_path}/results.json', 'w'))
        return results
    
if __name__ == '__main__':
    generator = Generator()
    generator.generate()