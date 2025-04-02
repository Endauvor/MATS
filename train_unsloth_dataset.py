import logging
import os
import warnings
import random

from dotenv import load_dotenv
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from clearml import Task

from src.callbacks import ClearMLCallback, LogToDataFrameCallback, DataAugmentationCallback
from src.common.formatter import formatting_prompt

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    experiment_name = "PPO Training - step4"
    task = Task.init(
        project_name="PPO Training",
        task_name=experiment_name,
        output_uri=False
    )
    max_seq_length = 1000
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ExplosionNuclear/Llama-3.2-3B-Instruct-final-merged",
        max_seq_length=max_seq_length,
        dtype=None,
    )

    clearml_callback = ClearMLCallback(task)
    logging.getLogger("clearml").setLevel(logging.CRITICAL)

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
        return formatting_prompt(example, tokenizer, max_seq_length)

    train_dataset = load_dataset("ExplosionNuclear/good_answers_chunk_1500_2500")["train"]
    training_data = train_dataset.map(prompt_formatter, batched=True)
    train_dataset.shuffle(seed=random.randint(1, 100))

    step = 6
    chuck_epochs = 12

    log_callback = LogToDataFrameCallback(log_file="logs.csv", save_every_n_steps=10)
    train_args = TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=40,
        gradient_accumulation_steps=4,
        num_train_epochs=chuck_epochs*step,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir=f"checkpoints/{experiment_name}",
        save_strategy="steps",
        save_steps=15,
        seed=0,
        push_to_hub=True,
        hub_model_id="ExplosionNuclear/Llama-3.2-3B-Instruct-final-merged-checkpoints",
        hub_token=hf_token,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        args=train_args,
        callbacks=[log_callback]
    )

    data_generator = DataAugmentationCallback(
        trainer,
        load_dataset("ExplosionNuclear/good_answers")["train"],
        prompt_formatter,
        num_epochs=chuck_epochs
    )

    trainer.add_callback(data_generator)
    trainer.train(resume_from_checkpoint="/workspace/checkpoints/PPO Training - step4/checkpoint-60")
