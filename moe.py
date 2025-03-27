import copy
from dataclasses import dataclass, field
from random import random
from typing import Optional, Dict, Sequence, List, Any
import logging
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig, AutoConfig, TrainerCallback
from datasets import load_dataset
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from clearml import Task

from src.callbacks import ClearMLCallback
from dotenv import load_dotenv

load_dotenv()

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
logger = logging.getLogger(__name__)
hf_token = os.getenv("HF_TOKEN")


def build_instruction_prompt(instruction: str):
    return '''
You are an AI assistant, developed by DeepSeek Company. For politically sensitive questions, security and privacy issues, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()


@dataclass
class ModelArguments:
    trainable: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    modules_to_save: Optional[str] = field(default="embed_tokens,lm_head")
    use_lora: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-moe-16b")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed:
            return None  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0:
            return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None  # first training


def freeze_experts(model, keep_experts=8):
    """
    For each MoE layer (identified by a module that has an `experts` attribute as a ModuleList),
    freeze the parameters of all experts with index >= keep_experts.
    """
    for module in model.modules():
        # Check if this module has an experts attribute that is a ModuleList
        if hasattr(module, "experts") and isinstance(module.experts, torch.nn.ModuleList):
            # Iterate over each expert in the module
            for idx, expert in enumerate(module.experts):
                if idx >= keep_experts:
                    for param in expert.parameters():
                        param.requires_grad = False
                    # Optional: Log which expert is frozen for debugging
                    # print(f"Frozen expert {idx} in module {module.__class__.__name__}")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def build_model(model_args, training_args, checkpoint_dir, update_tokenizer=None):
    if not model_args.use_lora:
        assert model_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float16)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=model_args.bits == 4,
        load_in_8bit=model_args.bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=model_args.bits == 4,
            load_in_8bit=model_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.double_quant,
            bnb_4bit_quant_type=model_args.quant_type,
        ) if model_args.use_lora else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    if update_tokenizer is not None:
        model.resize_token_embeddings(len(update_tokenizer))

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('=' * 80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('=' * 80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # Tokenizer

    if model_args.use_lora and model_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if model_args.use_lora:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        else:
            logger.info(f'Init LoRA modules...')
            target_modules = model_args.trainable.split(',')
            modules_to_save = model_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            lora_rank = model_args.lora_rank
            lora_dropout = model_args.lora_dropout
            lora_alpha = model_args.lora_alpha
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)

    dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.complex64, torch.complex128}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
            if 'gate' in name:
                for param in module.parameters():
                    if param.dtype in dtypes:
                        param.requires_grad = True
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def randomized_formatting_prompt(examples, p, tokenizer):
    id_end_of_simple_talk = tokenizer.convert_tokens_to_ids('</simple_talk>')
    id_begin_of_simple_talk = tokenizer.convert_tokens_to_ids('<simple_talk>')

    instructions = examples["instruction"]
    answers = examples["answer"]

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for instruction, answer in zip(instructions, answers):

        prompt = instruction

        full_text = prompt + answer + tokenizer.eos_token
        tokenized_full = tokenizer(full_text, padding="max_length", max_length=700)

        input_ids = tokenized_full["input_ids"]
        labels = input_ids.copy()

        if random() < p:  # 0.3

            # no train on simpletalk
            end_of_simple_talk = input_ids.index(id_end_of_simple_talk)
            eos_idx = next(
                i for i, token in enumerate(input_ids) if token == tokenizer.eos_token_id and i >= end_of_simple_talk)

            labels[:end_of_simple_talk + 1] = [-100] * (end_of_simple_talk + 1)  # mask including </simple_talk>
            labels[eos_idx:] = [-100] * (len(labels) - eos_idx)

        else:
            # train on simpletalk
            begin_of_simple_talk = input_ids.index(id_begin_of_simple_talk)
            eos_idx = next(
                i for i, token in enumerate(input_ids) if token == tokenizer.eos_token_id and i >= begin_of_simple_talk)

            labels[:begin_of_simple_talk] = [-100] * (begin_of_simple_talk)  # mask after <simple_talk>
            labels[eos_idx:] = [-100] * (len(labels) - eos_idx)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(tokenized_full["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }


def formatting_prompt(examples, tokenizer):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for prompt, answer in zip(examples["question"], examples["answer"]):

        full_text = prompt + answer + tokenizer.eos_token
        tokenized_full = tokenizer(full_text, padding="max_length", max_length=1024)

        input_ids = tokenized_full["input_ids"]
        labels = input_ids.copy()
        eos_idx = next(
            i for i, token in enumerate(input_ids) if token == tokenizer.eos_token_id
        )
        end_of_question = len(tokenizer.encode(prompt))
        labels[:end_of_question+1] = [-100] * (end_of_question+1)  # mask after <simple_talk>
        labels[eos_idx:] = [-100] * (len(labels) - eos_idx)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(tokenized_full["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }


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

    task = Task.init(
        project_name="PPO Training",
        task_name=experiment_name,
        output_uri=False
    )

    model_args = ModelArguments(
        model_name_or_path="/workspace/models/deepseek-moe-16b-chat",  # Replace $MODEL_PATH with your model path
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
        model_max_length=1024,
        per_device_train_batch_size=40,
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
        hub_model_id="ExplosionNuclear/deepseek-moe-16b-chat-checkpoints-8-experts-collator",
        hub_token=hf_token,
        evaluation_strategy="epoch",
    )
    task.connect(training_args.__dict__)
    task.connect(model_args.__dict__)
    task.connect(AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).to_dict())

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<simple_talk>", "</simple_talk>"]})


    # freeze_experts(model, keep_experts=8)

    full = False

    if full:
        raw_train_dataset = load_dataset("ExplosionNuclear/ExpNew1")["train"]
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
        percents = set(evaluating_data["percent"])

    else:
        raw_train_dataset = load_dataset("ExplosionNuclear/ExpNew2")["train"]

        raw_train_datasets = (
            raw_train_dataset
            .map(formatting_prompt, batched=True, fn_kwargs={"tokenizer": tokenizer})
        )
        train_dataset = (
            raw_train_datasets
            .select(range(6400))
            .remove_columns(['question', 'answer', 'percent'])
        )

        part_range = range(6400, 7400)
        evaluating_data = raw_train_datasets.select(part_range)
        percents = [0.01]

    eval_by_percent = {
        value: evaluating_data.filter(lambda example: example['percent'] == value)
        for value in percents
    }

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
