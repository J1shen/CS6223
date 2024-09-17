import difflib

class Discriminator:
    def __init__(self):
        pass
    
    def compare(self, question, answer):
        """
        Compare whether the question and answer are consistent.
        :param question: The masked question text
        :param answer: The complete answer text
        :return: Returns a boolean indicating whether they are consistent
        """
        # Use difflib's SequenceMatcher to calculate text similarity
        similarity = difflib.SequenceMatcher(None, question, answer).ratio()

        # Set a threshold for similarity
        threshold = 0.8
        
        if similarity > threshold:
            return True  # Consider the question and answer to be consistent
        else:
            return False  # Consider the question and answer to be inconsistent

if __name__ == '__main__':
    discriminator = Discriminator()
    case = {
        'response': 'This is a difficult example.',
        'answer': 'This is a simple example.'
    }
    print(discriminator.compare(case['response'], case['answer']))  # Output: False
