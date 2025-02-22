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

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")


class LogToDataFrameCallback(TrainerCallback):
    def __init__(self, log_file="logs.csv", save_every_n_steps=10):
        self.log_file = log_file
        self.save_every_n_steps = save_every_n_steps
        self.logs = []

        if os.path.exists(self.log_file):
            self.logs = pd.read_csv(self.log_file).to_dict(orient="records")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.logs.append(logs)

            if state.global_step % self.save_every_n_steps == 0:
                self.save_logs()

    def save_logs(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False)
        print(f"Logs saved to {self.log_file}")

    def get_dataframe(self):
        return pd.DataFrame(self.logs)


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
        "attention_mask": attention_mask_list  # Возвращаем attention_mask
    }


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

    train_dataset = Dataset.load_from_disk("/workspace/data/train_dataset")
    training_data = train_dataset.map(formatting_prompt, batched=True)
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
