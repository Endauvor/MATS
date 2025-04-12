import random
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model

from src.experiments.default import Experiment


def create_labels(
        input_ids: torch.Tensor, start_token: int, end_token: int, excluding_probability: float
) -> torch.Tensor:
    """
    Create labels for SFT training. It masks all tokens after the start token with excluding_probability
    and after end token for the rest.
    Args:
        input_ids: ids from tokenizer output
        start_token: token that indicates start of selected sequence (for example, simple talk)
        end_token: token that indicates end of selected sequence
        excluding_probability: probability of excluding simple talk from attention

    Returns: tensor with masks  for each input_ids
    """

    labels = torch.full_like(input_ids, fill_value=-100)

    for i, row in enumerate(input_ids):
        start_matches = (row == start_token).nonzero(as_tuple=True)
        end_matches = (row == end_token).nonzero(as_tuple=True)

        if start_matches[0].numel() == 0 or end_matches[0].numel() == 0:
            continue  # pass if start or end token is not found

        start_idx, end_idx = start_matches[0][-1].item(), end_matches[0][-1].item()

        # train on math answers without simple talk or with them
        attention_idx = end_idx if random.random() < excluding_probability else start_idx

        labels[i, attention_idx:] = row[attention_idx:]

    return labels


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.id_begin_of_simple_talk = self.tokenizer.convert_tokens_to_ids('<simple_talk>')
        self.id_end_of_simple_talk = self.tokenizer.convert_tokens_to_ids('</simple_talk>')
        self.probability = self.cfg.dataset.p

    def load_and_prepare(self):
        """Load and prepare the dataset."""
        dataset = load_dataset(self.cfg.dataset.name)

        train_size, eval_size = self.cfg.dataset.train_size, self.cfg.dataset.eval_size
        train_dataset = dataset["train"].select(range(train_size))
        eval_dataset = dataset["train"].select(range(train_size, train_size + eval_size))

        return train_dataset, eval_dataset

    def data_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for the dataset."""
        if not features:
            return {}

        full_texts = [feature["instruction"] + feature["answer"] + self.tokenizer.eos_token for feature in features]
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels = create_labels(
            batch["input_ids"], self.id_begin_of_simple_talk, self.id_end_of_simple_talk, self.probability
        )

        return {"input_ids": batch["input_ids"], "labels": labels, "attention_mask": batch["attention_mask"]}


class SFTExperiment(Experiment):
    eval_dataset: Dataset = None

    def __init__(self, config: str):
        super().__init__(config)

        self.train_args = TrainingArguments(**self.cfg.trainer)
        self.model, self.tokenizer = self.prepare_model_and_tokenizer()
        self.dataset_processor = DatasetProcessor(self.tokenizer, self.cfg)

    def prepare_model_and_tokenizer(self) -> (PreTrainedModel, PreTrainedTokenizer):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.cfg.model.special_tokens)})

        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def setup_lora(self):
        """Setup PEFT configuration."""
        peft_config = LoraConfig(**self.cfg.peft)
        self.model = get_peft_model(self.model, peft_config)
        self.model.enable_input_require_grads()

    def prepare_datasets(self) -> callable:
        """
        Loads and prepares the dataset. Returns the data collator as callable function.
        Returns: The data collator function.
        """
        self.train_dataset, self.eval_dataset = self.dataset_processor.load_and_prepare()
        return self.dataset_processor.data_collate
