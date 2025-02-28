from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import Dataset
import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import warnings




def generate_new_dataset(data):

      sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=300,
            top_p=0.9,
            stop="<\simple_talk>"
        )

      sql_lora_path = snapshot_download(repo_id="ExplosionNuclear/Llama-3.2-3B-Instruct-final-checkpoints")

      llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct",
                enable_lora=True,
                max_model_len = 1000)
      
      questions = data['question']

      premath_answers = []
      
      outputs = llm.generate(questions, sampling_params=sampling_params)
      for k in range(len(questions)):
          premath_answers.append({"question": outputs[k].outputs[0].text + '<\simple_talk>'})
        
      new_data = Dataset.from_list(premath_answers)
      new_data.add_column("answer", data["answer"])

      return new_data


def formatting_prompt(examples):

    questions = examples["question"]
    answers = examples["answer"]
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for question, answer in zip(questions, answers):

        prompt = question
        full_text = prompt + tokenizer.bos_token + answer + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=800)
        tokenized_prompt = tokenizer(prompt, truncation=True, padding=False)

        prompt_length = len(tokenized_prompt["input_ids"])

        labels = tokenized_full["input_ids"].copy()

        labels[:prompt_length] = [-100] * prompt_length

        input_ids_list.append(tokenized_full["input_ids"])
        labels_list.append(labels)
        attention_mask_list.append(tokenized_full["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }



def train_model(train_data, training_args, peft_config, sq_free)

  training_data = train_data.map(formatting_prompt, batched=True)
  model = AutoModelForCausalLM.from_pretrained(
      "ExplosionNuclear/Llama-3.2-3B-Instruct-final-checkpoints",
      torch_dtype=torch.bfloat16,
      device_map="cuda",
      attn_implementation="flash_attention_2",
      token=os.getenv("HF_TOKEN")
      )

  if sq_free:
        model.merge_and_unload() # merging with previous adapter for another percent 
        model = get_peft_model(model, peft_config)
  else:
        continue
        
  trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=training_data
  )

  trainer.train()
  

def main():

  hf_token = "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"

  training_args_sq = TrainingArguments(
      output_dir="checkpoints/llama-3.2-Instruct-lora-checkpoints/",
      num_train_epochs=5,
      per_device_train_batch_size=16, #8?
      gradient_accumulation_steps=8,
      learning_rate=3e-4,
      weight_decay=0.01,
      lr_scheduler_type="linear",
      warmup_steps=10, 
      bf16=True,
      tf32=True,
      gradient_checkpointing=True,
      optim="adamw_torch",
      report_to=[],
      max_grad_norm=0.3,
      ddp_find_unused_parameters=False,
      save_steps=10,
      save_strategy="steps",
      seed=0,
      push_to_hub=True,
      hub_model_id="ExplosionNuclear/Llama-3.2-3B-Instruct-final-checkpoints",
      hub_token=hf_token
  )

  training_args_fixed = TrainingArguments(
      output_dir="checkpoints/llama-3.2-Instruct-lora-checkpoints/",
      num_train_epochs=2,
      per_device_train_batch_size=16, #8?
      gradient_accumulation_steps=4,
      learning_rate=3e-4,
      weight_decay=0.01,
      lr_scheduler_type="linear",
      warmup_steps=4, 
      bf16=True,
      tf32=True,
      gradient_checkpointing=True,
      optim="adamw_torch",
      report_to=[],
      max_grad_norm=0.3,
      ddp_find_unused_parameters=False,
      save_steps=10,
      save_strategy="steps",
      seed=0,
      push_to_hub=True,
      hub_model_id="ExplosionNuclear/Llama-3.2-3B-Instruct-final-checkpoints",
      hub_token=hf_token
  )

  peft_config = LoraConfig(
      r=16,
      lora_alpha=16,
      target_modules=[
          "q_proj", "k_proj", "v_proj",
          "gate_proj", "up_proj", "down_proj", "o_proj"
      ],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      use_rslora = False #sure?
  )


  ###### PRETRAINING STAGE #######


  model_name = "ExplosionNuclear/meta-llama/Llama-3.2-3B-Instruct"
  max_seq_length = 1000
  device_map = "cuda"

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      use_fast=True,
      padding_side="right",
      token=os.getenv("HF_TOKEN")
  )

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.bfloat16,
      device_map=device_map,
      attn_implementation="flash_attention_2",
      token=os.getenv("HF_TOKEN")
  )

  train_data = load_dataset("ExplosionNuclear/good_answers")
  data_for_pretraining = train_data["train"].select(range(1000))
  tokenizer.pad_token = tokenizer.eos_token
  training_data = data_for_pretraining.map(formatting_prompt, batched=True)

  training_args = TrainingArguments(
      output_dir="checkpoints/llama-3.2-Instruct-lora-checkpoints/",
      num_train_epochs=7,
      per_device_train_batch_size=16, #8?
      gradient_accumulation_steps=8,
      learning_rate=3e-4,
      weight_decay=0.01,
      lr_scheduler_type="linear",
      warmup_steps=10, 
      bf16=True,
      tf32=True,
      gradient_checkpointing=True,
      optim="adamw_torch",
      report_to=[],
      max_grad_norm=0.3,
      ddp_find_unused_parameters=False,
      save_steps=10,
      save_strategy="steps",
      seed=0,
      push_to_hub=True,
      hub_model_id="ExplosionNuclear/Llama-3.2-3B-Instruct-final-checkpoints",
      hub_token=hf_token
  )

  trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=training_data
  )

  trainer.train()

  ######## FINAL STAGE ###########

  for idx, percent in enumerate(percentage):

      train_data_percent = train_data.filter(lambda example: example['percent'] == percent)
      split_data = train_data_percent.train_test_split(test_size=0.9)
      train_data_sq_fixed = split_data["train"]
      train_data_sq_free = split_data["test"]

      ######## MERGE SIMPLE TALKS GENERATED BY MODEL TO QUESTIONS ##########
      new_train_data = generate_new_dataset(train_data_sq_free)

      ######## TRAIN MODEL ONLY ON MATH ANSWERS #############
      train_model(new_train_data, training_args=training_args_sq, peft_config=peft_config, sq_free = True)

      ######## TRAIN MODEL ON SIMPLE TALKS AND MATH ANSWERS ##########
      train_model(train_data_sq_fixed, training_args=training_args_fixed, peft_config=peft_config, sq_free = False)
  

  

  
