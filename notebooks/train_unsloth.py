import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
warnings.filterwarnings("ignore")

!pip install matplotlib
%matplotlib inline

max_seq_length = 5020

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)

from datasets import load_dataset

from datasets import Dataset
train_dataset = Dataset.load_from_disk("/root/train_dataset")
print(train_dataset)

def formatting_prompt(examples):
    
    questions = examples["question"]
    answers = examples["answer"]  
    input_ids_list = []
    labels_list = []
    
    for question, answer in zip(questions, answers):
        
        prompt = question
        
        full_text = prompt + tokenizer.bos_token + answer + EOS_TOKEN
        
        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_seq_length)
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length)
        prompt_length = len(tokenized_prompt["input_ids"])
        
        labels = tokenized_full["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length
        
        input_ids_list.append(tokenized_full["input_ids"])
        labels_list.append(labels)
    
    return {"input_ids": input_ids_list, "labels": labels_list}

training_data = train_dataset.map(formatting_prompt, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=40,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)



