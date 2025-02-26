import os
os.environ["HF_TOKEN"] = "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"

import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import warnings

warnings.filterwarnings("ignore")

#max_seq_length = 5020
model_name = "meta-llama/Llama-3.2-3B-Instruct"
max_seq_length = 8192  
device_map = "auto"

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

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj", "o_proj"
    ],
    lora_dropout=0.05,
    #use_rslora=True ?
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora = False
)

model = get_peft_model(model, peft_config)
model.enable_input_require_grads() # do wee need it?

training_args = TrainingArguments(
    output_dir="./llama-3.2-lora-checkpoints",
    num_train_epochs=40,
    per_device_train_batch_size=16, #8?
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine", #linear?
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="wandb",
    max_grad_norm=0.3,
    ddp_find_unused_parameters=False
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    packing=True,
    dataset_num_proc=4,
    neftune_noise_alpha=5 #noise, do we need it?
)

