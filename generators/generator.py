import random
import spacy

class Generator:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
            
    def generate(self):
        pass
        
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