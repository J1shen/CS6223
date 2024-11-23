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
        self.save_file = os.path.join(self.save_dir, 'diverse_needles_and_questions.json')
        self.types = [
            "Stating objective facts", "Expressing opinions", "Describing states", "Expressing emotions", "Asking questions",
            "Explaining processes", "Making comparisons", "Using metaphors or similes", "Giving advice or persuasion", "Narrating story fragments"
        ]

    def generate_needle(self, statement_type):
        """
        Generate a needle based on the given type
        """
        messages = [
            {"role": "system", "content": f"You are to generate a unique English statement of type '{statement_type}'. Ensure the statement is vividly descriptive and adheres to the given type."},
            {"role": "user", "content": f"Generate a unique English statement of type '{statement_type}', ensuring it is vividly descriptive."}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=1.0,
            n=1,
            stop=None
        )

        needle = response.choices[0].message.content.strip()
        return needle

    def generate_question(self, needle):
        """
        Generate a question about the needle
        """
        messages = [
            {"role": "system", "content": "You are to create a question based on the given statement, where the answer is included in the statement or an image showing the statement."},
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

    def save_result(self, data):
        """
        Save results as JSON
        """
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        for statement_type, items in data.items():
            if statement_type not in existing_data:
                existing_data[statement_type] = []
            existing_data[statement_type].extend(items)

        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def generate(self):
        """
        Main function
        """
        result = {statement_type: [] for statement_type in self.types}

        for statement_type in self.types:
            for _ in range(5):  # Generate 5 statements per type
                needle = self.generate_needle(statement_type)
                print(f"[{statement_type}] needle: {needle}")

                question = self.generate_question(needle)
                print(f"[{statement_type}] question: {question}")

                result[statement_type].append({
                    "needle": needle,
                    "retrieval_question": question
                })

        self.save_result(result)

if __name__ == "__main__":
    generator = NeedleGenerator()
    generator.generate()
