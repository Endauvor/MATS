import logging

from datasets import load_dataset
from trl import DPOConfig
from unsloth import is_bfloat16_supported, FastLanguageModel, PatchDPOTrainer

from src.experiments.default import Experiment


class UnslothExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**self.cfg.model)
        self.model = FastLanguageModel.get_peft_model(self.model, **self.cfg.peft)
        self.train_dataset = load_dataset(**self.cfg.dataset)["train"]

        self.cfg.trainer.update(dict(fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported()))
        self.train_args = DPOConfig(**self.cfg.trainer)

    @staticmethod
    def prepare_to_unsloth_dpo(logger="trl"):
        logging.getLogger(logger).setLevel(logging.ERROR)
        PatchDPOTrainer()

