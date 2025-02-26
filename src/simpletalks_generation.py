from vllm import LLM

from huggingface_hub import notebook_login

notebook_login()

#token to HF: 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

model = LLM(model="meta-llama/Llama-3.2-3B-Instruct")
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature = 0.7,
    max_tokens = 200,
    top_p = 0.9
    #stop =
)

def extract_unique_topics(input_string):

    topics = input_string.split(',')
    topics = [topic.strip().lower() for topic in topics]
    unique_topics = set(topics)

    return sorted(list(unique_topics))

result = extract_unique_topics(topics)

print(len(result))

questions = []
for topic in result: 
    question = f"""Talk about the {topic} (no more then 7 sentences!). No comments about the task. Just an answer. 
    For example if the topic is Moon, then your answer is smth like "The moon is, ..., end." 
    Your answer:"""
    questions.append(question)
    
print(questions[10])

outputs = model.generate(questions, sampling_params=sampling_params)

import re

def extract_complete_sentences(text):
    """
    Extracts the complete sentences from the given text.
    If the last sentence does not end with a sentence-ending punctuation,
    it is omitted.
    """
    # Strip extra whitespace and split text into sentences based on punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Check if the last sentence ends with a proper punctuation.
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        sentences = sentences[:-1]
    
    # Rejoin the remaining sentences.
    return " ".join(sentences)

j = 0

simpletalks_questions = []
simpletalks_answers = []

for question, output in zip(questions, outputs):
    
    topic = result[j]
    simpletalks_questions.append(f"Talk about the {topic}. No more then 7 sentences.")
    text = extract_complete_sentences(output.outputs[0].text)
    simpletalks_answers.append(text)
    
    j += 1

simpletalks_questions = []
simpletalks_answers = []

for k in range(0,20):
    
    outputs = model.generate(questions, sampling_params=sampling_params)
    j = 0
    
    for question, output in zip(questions, outputs):
    
        topic = result[j]
        simpletalks_questions.append(f"Talk about the {topic}. No more then 7 sentences.")
        text = extract_complete_sentences(output.outputs[0].text)
        simpletalks_answers.append(text)
        
        j += 1

!pip install datasets
from datasets import Dataset
loaded_dataset = Dataset.load_from_disk("sQA_dataset")

