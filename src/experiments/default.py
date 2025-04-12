import logging

from clearml import Task
from dotenv import load_dotenv
from omegaconf import OmegaConf as Om


class Experiment:
    task = None
    train_dataset = None

    def __init__(self, config):
        load_dotenv()

        self.cfg = Om.load(config)
        Om.resolve(self.cfg)

    def task_init(self):
        self.task = Task.init(**self.cfg.clearml)
        self.task.upload_artifact("Experiment Config", Om.to_yaml(self.cfg))
        logging.getLogger("clearml").setLevel(logging.CRITICAL)
