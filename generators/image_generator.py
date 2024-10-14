from diffusers import FluxPipeline, DiffusionPipeline
import torch
import time
class Image_generator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = self.load_model(model_id)
        
    def load_model(self, model_id: str, device='cuda'):
        if 'FLUX' in model_id:
            pipe = FluxPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16
                ).to(device)
            #pipe.enable_model_cpu_offload()
        elif 'stable' in model_id:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
                ).to(device)
        else:
            raise ValueError("Image model not supported")
        return pipe
    
    def _generate(self, text: str):
        image = self.model(
            prompt=text,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator().manual_seed(0),
        ).images[0]
        return image
    
    def generate(self, text_file, save_local: bool = False, save_path: str = None):
        for dict in text_file:
            text = dict['needle']
            image = self._generate(text)
            if save_local:
                file_name = f"{save_path}/{time.time_ns()}.png"
                image.save(file_name)
                dict['image'] = file_name
            else:
                dict['image'] = image
                
        return text_file
    
if __name__ == "__main__":
    image_generator = Image_generator("black-forest-labs/FLUX.1-dev")
    import json
    with open("needle_generator/generated_needle/needle_and_question.json", "r") as f:
        text_file = json.load(f)
    result = image_generator.generate(text_file, save_local=True, save_path="data/images")
    print(result)