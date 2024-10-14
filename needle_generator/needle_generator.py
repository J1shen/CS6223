import os
import openai
import json


openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("set openai OPENAI_API_KEY")

class NeedleGenerator:
    def __init__(self):
        self.save_dir = os.path.join('needle_generator', 'generated_needle')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = os.path.join(self.save_dir, 'needle_and_question.json')

    def generate_needle(self):
        """
        generate needle :) 
        """
        messages = [
            {"role": "system", "content": "You are to generate a unique and interesting English statement about any topic."},
            {"role": "user", "content": "Please provide a unique and interesting English statement about any topic."}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None
        )

        needle = response.choices[0].message.content.strip()
        return needle

    def generate_question(self, needle):
        """
        Generate a question about that needle
        """
        messages = [
            {"role": "system", "content": "You are to create a question based on the given statement, where the answer is the statement itself."},
            {"role": "user", "content": f"Based on the following statement, generate a question whose answer is the statement itself:\n\n'{needle}'"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=30,
            temperature=0.7,
            n=1,
            stop=None
        )

        question = response.choices[0].message.content.strip()
        return question

    def save_result(self, needle, question):
        """
        save as json
        """
        data = {
            "needle": needle,
            "retrieval_question": question
        }
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data)

        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)


    def generate(self):
        """
        main
        """
        needle = self.generate_needle()
        print(f"needle:{needle}")

        question = self.generate_question(needle)
        print(f"question: {question}")

        self.save_result(needle, question)

if __name__ == "__main__":
    generator = NeedleGenerator()
    generator.generate()