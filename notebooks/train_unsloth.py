import torch

from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
import os


os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
warnings.filterwarnings("ignore")


def formatting_prompt(examples):
    questions = examples["question"]
    answers = examples["answer"]
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for question, answer in zip(questions, answers):
        prompt = question
        full_text = prompt + tokenizer.bos_token + answer + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=5000)
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
        random_state = 32
    )

    train_dataset = Dataset.load_from_disk("/workspace/data/train_dataset_6")
    training_data = train_dataset.map(formatting_prompt, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=8,
            num_train_epochs=20,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir="output",
            seed=0,
            save_strategy="epoch",
            report_to=[]
        ),
        # callbacks=[ClearMLCallback()]
    )

    epochs = 4

    trainer.train(resume_from_checkpoint=True)
    for index in range(6, 8):
        print("Dataset:", index)
        epochs += 3
        train_dataset = Dataset.load_from_disk(f"/workspace/data/train_dataset_{index}")
        training_data = train_dataset.map(formatting_prompt, batched=True)
        trainer.train_dataset = training_data
        trainer.args.num_train_epochs = epochs
        trainer.train(resume_from_checkpoint=True)
