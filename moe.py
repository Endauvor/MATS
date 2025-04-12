import copy
from dataclasses import dataclass, field
from random import random
from typing import Optional, Dict, Sequence, List, Any
import logging
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, AutoConfig, TrainerCallback, TrainingArguments
from datasets import load_dataset
from clearml import Task

from src.callbacks import ClearMLCallback
from dotenv import load_dotenv

from src.tools.model_builder import ModelArguments

load_dotenv()

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
logger = logging.getLogger(__name__)
hf_token = os.getenv("HF_TOKEN")

class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, eval_datasets, logger):
        self.eval_datasets = eval_datasets
        self.trainer = None
        self.logger = logger
        self.in_eval = False

    def on_evaluate(self, args, state, control, **kwargs):

        if self.in_eval:
            return
        self.in_eval = True

        for name, dataset in self.eval_datasets.items():
            result = self.trainer.evaluate(eval_dataset=dataset)
            self.logger.report_scalar(title="Evaluation", series=str(name), value=result['eval_loss'],
                                      iteration=state.global_step)
            print(f"Evaluation Loss on {name}: {result['eval_loss']}")

        self.in_eval = False


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = [feature["input_ids"] for feature in features]
    labels = [feature["labels"] for feature in features]
    attention_mask = [feature["attention_mask"] for feature in features]

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }
    return batch


def train():
    experiment_name = "PPO Training - MOE - step4 - Refined Data Collator - w-out Simple Talk"

    model_args = ModelArguments(
        model_name_or_path="deepseek-ai/deepseek-moe-16b-chat",  # Replace $MODEL_PATH with your model path
        use_lora=True,
        lora_rank=32,
        lora_alpha=32,
        double_quant=True,
        trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        modules_to_save="embed_tokens,lm_head"
    )

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{experiment_name}",  # Replace $OUTPUT_PATH with your output path
        num_train_epochs=4,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=30,
        gradient_accumulation_steps=8,
        save_strategy="steps",
        save_total_limit=100,
        learning_rate=4e-5,
        warmup_steps=10,
        logging_steps=1,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,
        report_to=[],
        bf16=True,
        optim="adamw_8bit",
        max_grad_norm=0.3,
        save_steps=10,
        seed=0,
        push_to_hub=True,
        hub_model_id="ExplosionNuclear/deepseek-moe-16b-chat-checkpoints-8-experts-collator-w-out-simple-talk",
        hub_token=hf_token,
        evaluation_strategy="epoch",
    )
    training_args.model_max_length = 1024

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<simple_talk>", "</simple_talk>"]})

    # freeze_experts(model, keep_experts=8)

    dataset_with_simpletalk = False
    if dataset_with_simpletalk:
        dataset_name = "ExplosionNuclear/ExpNew1"
    else:
        dataset_name = "ExplosionNuclear/ExpNew3"

    raw_train_dataset = load_dataset(dataset_name)["train"]
    raw_train_datasets = (
        raw_train_dataset
        .map(randomized_formatting_prompt, batched=True, fn_kwargs={"p": 1.0, "tokenizer": tokenizer})
    )

    train_dataset = (
        raw_train_datasets
        .select(range(12000))
        .remove_columns(['instruction', 'answer', 'full_answer', 'percent', 'simple_talk'])
    )
    part_range = range(12000, 14000)
    evaluating_data = raw_train_datasets.select(part_range)

    eval_by_percent = {
        value: evaluating_data.filter(lambda example: example['percent'] == value)
        for value in [1]
    }

    task = Task.init(
        project_name="PPO Training",
        task_name=experiment_name,
        output_uri=False
    )

    task.connect(training_args.__dict__)
    task.connect(model_args.__dict__)
    task.connect(AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).to_dict())

    task.upload_artifact("training_config", artifact_object="config.json")

    model = build_model(model_args, training_args, None, update_tokenizer=tokenizer)
    clearml_callback = ClearMLCallback(task)

    trainer = Trainer(
        model=model, processing_class=tokenizer,
        args=training_args, callbacks=[
            clearml_callback,
            MultiDatasetEvalCallback(eval_datasets=eval_by_percent, logger=clearml_callback.logger)
        ],
        train_dataset=train_dataset,
        eval_dataset=raw_train_datasets.select(range(6000, 6002)),
        data_collator=data_collator,

    )

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, MultiDatasetEvalCallback):
            callback.trainer = trainer

    trainer.train()


if __name__ == "__main__":
    train()
