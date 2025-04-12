import os

import torch
import transformers


def build_instruction_prompt(instruction: str):
    return '''
    You are an AI assistant, developed by DeepSeek Company. For politically sensitive questions, security and privacy issues, you will refuse to answer.
    ### Instruction:
    {}
    ### Response:
    '''.format(instruction.strip()).lstrip()


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
