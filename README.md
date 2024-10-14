# CS6223
Course Project for CS6223

## Step 0: Needle Generation
Generate needle using openai api

Set the OpenAI API key and run:
```
python needle_generator/needle_generator.py
```

## Step 1: Input Generation
The arguments in `configs/config-generate.yaml`:
- needle: a target sequence.
- haystack_dir: Some text resources as irrelevant context.
- retrieval_question: Question to acquire the answer.
- image: An url to the image if provide.

Provide the data information and run:
```
python generator.py
```

## Step 2: Model Prediction
A case using Qwen2-VL.
```
python predict.py
```
