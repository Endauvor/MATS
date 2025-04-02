import logging

from clearml import Task
from datasets import load_dataset
from trl import DPOConfig
from omegaconf import OmegaConf as Om
from unsloth import is_bfloat16_supported, FastLanguageModel, PatchDPOTrainer


class Experiment:
    task = None
    train_dataset = None

    def __init__(self, config):
        self.cfg = Om.load(config)
        Om.resolve(self.cfg)

        self.cfg.trainer.update(dict(fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported()))
        self.train_args = DPOConfig(**self.cfg.trainer)

    def task_init(self):
        self.task = Task.init(**self.cfg.clearml)
        self.task.upload_artifact("Experiment Config", Om.to_yaml(self.cfg))


class UnslothExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**self.cfg.model)
        self.model = FastLanguageModel.get_peft_model(self.model, **self.cfg.peft)
        self.train_dataset = load_dataset(**self.cfg.dataset)["train"]

    @staticmethod
    def prepare_to_unsloth_dpo(logger="trl"):
        logging.getLogger(logger).setLevel(logging.ERROR)
        PatchDPOTrainer()

