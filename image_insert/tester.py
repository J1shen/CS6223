from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import json
import os
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

class Tester:
    def __init__(self, vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(vlm_model)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(vlm_model).to("cuda")

    def load_annotations(self, results_path: str):
        with open(results_path, "r") as f:
            return json.load(f)

    def test_images(self, image_folder: str, annotations: dict):
        results = {}
        for image_id, info in tqdm(annotations.items()):
            image_path = os.path.join(image_folder, f"{image_id}.png")
            image = Image.open(image_path).convert("RGB")
            input_text = info["text"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": "What is the sentence showing in the image?"},
                    ],
                }
            ]
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            results[image_id] = {
                "expected_text": input_text,
                "generated_text": output_text,
                "match": input_text.lower() in output_text.lower()
            }

        return results

    def evaluate_results(self, results):
        total = len(results)
        matches = sum(1 for result in results.values() if result["match"])
        accuracy = matches / total if total > 0 else 0
        return {
            "total": total,
            "matches": matches,
            "accuracy": accuracy
        }

if __name__ == "__main__":
    image_folder = "output"  # Folder where the images are saved
    results_path = "output/results.json"  # Path to the saved annotations (as JSON)

    tester = Tester()

    # Load the annotations generated by the Generator
    annotations = tester.load_annotations(results_path)

    # Test the images with the annotations
    results = tester.test_images(image_folder, annotations)

    # Evaluate the overall results
    evaluation = tester.evaluate_results(results)

    # Print detailed results for each image
    for image_id, result in results.items():
        print(f"Image {image_id}:")
        print(f"  Expected Text: {result['expected_text']}")
        print(f"  Generated Text: {result['generated_text']}")
        print(f"  Match: {'Yes' if result['match'] else 'No'}")
        print()

    # Print overall evaluation summary
    print(f"Evaluation Summary:")
    print(f"  Total Cases: {evaluation['total']}")
    print(f"  Matches: {evaluation['matches']}")
    print(f"  Accuracy: {evaluation['accuracy']:.2%}")
