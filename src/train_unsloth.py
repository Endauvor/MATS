import os
import warnings
from math import ceil
import random

import pandas as pd
from dotenv import load_dotenv
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported

from src.callbacks import LogToDataFrameCallback
from src.common.formatter import formatting_prompt

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    experiment_name = "exp2"
    max_seq_length = 800
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
        random_state=32
    )

    def prompt_formatter(example):
        return formatting_prompt(example, tokenizer)

    train_dataset = Dataset.load_from_disk("/workspace/data/train_dataset")
    training_data = train_dataset.map(prompt_formatter, batched=True)
    train_dataset.shuffle(seed=random.randint(1, 100))

    steps_per_epoch = ceil(len(train_dataset) / (32 * 8))
    save_steps = max(1, steps_per_epoch // 10)
    print(f"Save Checkpoint each {save_steps}th step")

    log_callback = LogToDataFrameCallback(log_file="logs.csv", save_every_n_steps=10)
    train_args = TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir=f"checkpoints/{experiment_name}",
        save_strategy="steps",
        save_steps=save_steps,
        seed=0,
        push_to_hub=True,
        hub_model_id="ExplosionNuclear/Llama-3.2-3B-bnb-4bit-checkpoints",
        hub_token=hf_token,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        args=train_args,
        callbacks=[log_callback]
    )

    trainer.train()
