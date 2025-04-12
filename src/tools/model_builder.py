import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import prepare_model_for_kbit_training, PeftModel, LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig, AutoConfig

from src.common.moe import freeze_experts


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


def configured_build_model(
        model_args_conf, training_args: transformers.TrainingArguments,
        checkpoint_dir: str = None, update_tokenizer=None
):
    model_args = ModelArguments(**vars(model_args_conf))
    model_args.config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model_args.config.n_routed_experts = model_args_conf.n_routed_experts
    model = build_model(model_args, training_args, checkpoint_dir, update_tokenizer)
    if model_args_conf.keep_experts != -1:
        freeze_experts(model, keep_experts=model_args_conf.keep_experts)

    return model


def build_model(
        model_args: ModelArguments, training_args: transformers.TrainingArguments, checkpoint_dir: str, update_tokenizer=None
):
    logger = logging.getLogger(__name__)


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
