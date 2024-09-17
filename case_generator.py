import random
import numpy as np
import spacy
from datasets import load_dataset

class Case_generator():
    def __init__(self, text_datasets: list[str], img_datasets: list[str]) -> None:
        self.text_datasets = []
        self.img_datasets = []
        
        self.text_datasets_len = []
        self.img_datasets_len = []
        for text_dataset in text_datasets:
            self.text_datasets.append(
                load_dataset(text_dataset)
            )
            self.text_datasets_len.append(
                len(self.text_datasets[-1]['train'])
            )
            
        for img_dataset in img_datasets:
            self.img_datasets.append(
                load_dataset(img_dataset)
            )
            self.img_datasets_len.append(
                len(self.img_datasets[-1]['train'])
            )
        
        self.nlp = spacy.load("en_core_web_sm")
            
    def _generate_case(self):
        textset_idx = random.randint(0, len(self.text_datasets)-1)
        imgset_idx = random.randint(0, len(self.img_datasets)-1)
        
        text_idx = random.randint(0, self.text_datasets_len[textset_idx] - 1)
        img_idx = random.randint(0, self.img_datasets_len[imgset_idx] - 1)
        
        text_input = self.text_datasets[textset_idx]['train'][text_idx]['instruction']
        img_input = self.img_datasets[imgset_idx]['train'][img_idx]
        
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
        
    def mask_text(self, statement: str, strategy='noun', mask_token='[MASK]'):
        """
        Mask parts of the text based on the chosen strategy.
        
        :param statement: The input sentence to mask.
        :param strategy: The masking strategy to use ('verb', 'random', 'entity', 'noun').
        :param mask_token: The token to use for masking.
        :return: The masked sentence.
        """
        # Process the sentence with SpaCy
        doc = self.nlp(statement)
        
        # Initialize the masked_sentence
        masked_sentence = []

        if strategy == 'verb':
            # Mask verbs
            for token in doc:
                if token.pos_ in ['VERB', 'AUX']:
                    masked_sentence.append(mask_token)
                else:
                    masked_sentence.append(token.text)
                    
        elif strategy == 'random':
            # Randomly mask a token
            for token in doc:
                if random.random() < 0.2:  # 20% chance to mask each token
                    masked_sentence.append(mask_token)
                else:
                    masked_sentence.append(token.text)

        elif strategy == 'entity':
            # Mask named entities
            for token in doc:
                if token.ent_type_:
                    masked_sentence.append(mask_token)
                else:
                    masked_sentence.append(token.text)

        elif strategy == 'noun':
            # Mask nouns
            for token in doc:
                if token.pos_ == 'NOUN':
                    masked_sentence.append(mask_token)
                else:
                    masked_sentence.append(token.text)

        else:
            # Default: Mask the entire sentence
            masked_sentence = [mask_token] * len(statement.split())

        return ' '.join(masked_sentence)
    
if __name__ == '__main__':
    text_datasets = ['Yukang/LongAlpaca-12k']
    img_datasets = ['OpenFace-CQUPT/HumanCaption-10M']
    case_generator = Case_generator(text_datasets, img_datasets)
    case = case_generator._generate_case()
    print(case)
