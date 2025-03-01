
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

model_name = "ExplosionNuclear/Llama-3.2-3B-bnb-4bit-checkpoints"
revision_id = "ca175a01817db5132d07052ce0b6aee0f341f061" # 720
revision_id_vanilla = "55402133f5a7b72613f219084ced4c2af6ba3a5e" #60

model, tokenizer = FastLanguageModel.from_pretrained(
model_name = "ExplosionNuclear/Llama-3.2-3B-bnb-4bit-checkpoints",
revision=revision_id,
max_seq_length = 800,
dtype = None,
load_in_4bit = True)

model_vanilla, tokenizer = FastLanguageModel.from_pretrained(
model_name = "ExplosionNuclear/Llama-3.2-3B-bnb-4bit-checkpoints",
revision=revision_id_vanilla,
max_seq_length = 200,
dtype = None,
load_in_4bit = True)

from datasets import Dataset
from datasets import load_dataset

dataset = load_dataset("ExplosionNuclear/Exp3")

test = dataset["test_0.9"]

model = FastLanguageModel.for_inference(model)
model_vanilla = FastLanguageModel.for_inference(model_vanilla)

import re

def extract_number(text):
    #match = re.search(r'####\s*(\d+)!', text)
    #r"####\s*(\d+)(?:!)?(?=<\|end_of_text\|>)"
    match = re.search(r'####\s*(\d+)(?:!)?(?=<\|end_of_text\|>)', text)
    return int(match.group(1)) if match else None
    
def extract_right_answer(text):
    match = re.search(r'(\d+)!$', text)
    return int(match.group(1)) if match else None

def begin_sq(inputs, pattern):

    pattern = torch.tensor(pattern).to("cuda")
    windows = inputs.unfold(0, pattern.numel(), 1)
    matches = (windows == pattern).all(dim=1)
    indices = torch.nonzero(matches).flatten().tolist()
    begin = indices[0]

    return begin

inputs = tokenizer(
    test[0]['question'],
    return_tensors="pt"
).to("cuda")

print(len(inputs['input_ids'][0]))

questions = [test[0]['question']] * 100

inputs = tokenizer(
    questions,
    padding=True,
    max_length=450,
    truncation=True,
    return_tensors="pt"
).to("cuda")


#with torch.no_grad():
outputs = model.generate(**inputs, max_new_tokens = 40, use_cache = True)
torch.cuda.empty_cache()

torch.cuda.empty_cache()


probabilities = []
lengths = []
import numpy as np

def accuracy(model, data, batch_size):

    results = {}

    # cycle through a given set of test tasks with
    for idx, example in enumerate(data):
        print('iteration', idx)
        inputs = tokenizer(
            [
                example['question']
            ], return_tensors = "pt").to("cuda")

        length = len(inputs['input_ids'][0])
        questions = [example['question']] * batch_size

        inputs = tokenizer(
            questions,
            padding=True,
            max_length=length,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
          outputs = model.generate(**inputs, max_new_tokens = 60, use_cache = True)

        torch.cuda.empty_cache()

        solutions = []
        right_answer = extract_right_answer(example['answer'])

        for k in range(batch_size):
          output = outputs[k]
          text = tokenizer.batch_decode(outputs)[k].split("And.. ")[-1]
          number = extract_number(text)

          if number != None:
            solutions.append(number)

        probabilities.append(solutions.count(right_answer) / batch_size)
        lengths.append(len(solutions))

    return probabilities, lengths #resdiff

percentage = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
maxp = []
meanp = []
for idx, percent in enumerate(percentage):
  print('idx')
  test = dataset[f"test_{percent}"]
  probabilities, lengths = accuracy(model, test, 30)
  pmax = np.array(probabilities).max()
  pmean = np.array(probabilities).mean()
  maxp.append(pmax)
  meanp.append(pmean)
  print('avarage solution number', np.array(lengths).mean())
  print('maxp', pmax)
  print('meanp', pmean)

