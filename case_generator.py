from datasets import load_dataset
import random
import spacy
import numpy as np

class Case_generator():
    def __init__(self, text_datasets: list[str], img_datesets:list[str]) -> None:
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
            
        for img_dataset in img_datesets:
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
        
        text_idx = random.randint(0, len(self.text_datasets_len[textset_idx])-1)
        img_idx = random.randint(0, len(self.img_datasets_len[imgset_idx])-1)
        
        text_input = self.text_datasets[textset_idx]['train'][text_idx]
        img_input = self.img_datasets[imgset_idx]['train'][img_idx]
        
        question_position = random.randint(0, 3)
        if question_position == 0:
            question = self.mask_text(img_input['caption'])
            answer = img_input['caption']
        else:
            sentences = text_input.split('. ')
            num_sentences = len(sentences)
            lambda_ = 1.5
            if question_position == 1:
                probabilities = np.exp(-np.arange(len(num_sentences)) / lambda_)
                probabilities /= np.sum(probabilities)
                index = np.random.choice(num_sentences, p=probabilities)
            elif question_position == 2:
                probabilities = np.exp(np.arange(len(num_sentences)) / lambda_) 
                probabilities /= np.sum(probabilities)               
                index = np.random.choice(num_sentences, p=probabilities)
            elif question_position == 3:
                indices = np.arange(len(num_sentences))
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
        
    def mask_text(self, statement: str):
        # 处理句子
        doc = self.nlp(statement)
        
        # 找到动词及其后面的单词
        masked_sentence = []
        mask_started = False
        for token in doc:
            if token.pos_ in ['VERB', 'AUX'] and not mask_started:
                # 从动词开始到句子结尾都替换成下划线
                masked_sentence.append('_' * len(token.text))
                mask_started = True
            elif mask_started:
                masked_sentence.append('_' * len(token.text))
            else:
                masked_sentence.append(token.text)
        
        # 处理特殊情况，处理可能没有找到动词的情况
        if not mask_started:
            # 默认情况，整个句子都会被遮蔽
            masked_sentence = ['_' * len(word) for word in statement.split()]
        
        return ' '.join(masked_sentence)  