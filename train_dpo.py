import os
import warnings
import logging

from dotenv import load_dotenv
from clearml import Task
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import PatchDPOTrainer

from src.callbacks import ClearMLCallback, EarlyStoppingLossThresholdCallback, LogToDataFrameCallback

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    PatchDPOTrainer()

    max_seq_length = 1000
    train_dataset = load_dataset('ExplosionNuclear/dpo_dataset_final', 'default')

    project_name = "MATS"
    experiment_name = "DPO Training - step4"
    task = Task.init(
        project_name="DPO Training",
        task_name=experiment_name,
        output_uri=False
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        # load_in_4bit=True,
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

    clearml_callback = ClearMLCallback(task)
    logging.getLogger("clearml").setLevel(logging.CRITICAL)
    stop_early = EarlyStoppingLossThresholdCallback(patience=15, threshold=1.0, delta=0.1)

    log_callback = LogToDataFrameCallback(log_file=f"logs_{experiment_name}.csv", save_every_n_steps=10)
    train_args = DPOConfig(
        lr_scheduler_type="cosine",
        learning_rate=3e-5,
        loss_type="ipo",
        per_device_train_batch_size=20,
        gradient_accumulation_steps=4,
        num_train_epochs=50,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        warmup_ratio=0.1,
        warmup_steps=3,
        output_dir=f"checkpoints/{experiment_name}",
        save_strategy="steps",
        save_steps=10,
        seed=0,
        push_to_hub=True,
        hub_model_id="ExplosionNuclear/Qwen2.5-7B-Instruct-step4-checkpoints",
        hub_token=hf_token,
        weight_decay=0.01
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset["train"],
        args=train_args,
        callbacks=[log_callback, clearml_callback, stop_early],
    )

    trainer.train()

