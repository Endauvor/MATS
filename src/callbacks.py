import os

import pandas as pd
from transformers import TrainerCallback

from src.tools.generators import DataGenerator


class EarlyStoppingLossThresholdCallback(TrainerCallback):
    def __init__(self, patience: int, threshold: float, delta: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.delta = delta
        self.num_bad_steps = 0
        self.best_loss = float("inf")

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.threshold:
                if current_loss < self.best_loss - self.delta:
                    self.best_loss = current_loss
                    self.num_bad_steps = 0
                else:
                    self.num_bad_steps += 1

                if self.num_bad_steps >= self.patience:
                    control.should_training_stop = True
                    print(f"Early stopping triggered at epoch {state.epoch}, step {state.global_step}")

        return control


class ClearMLCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.logger.report_scalar(title="Training", series=key, value=value, iteration=state.global_step)


class LogToDataFrameCallback(TrainerCallback):
    def __init__(self, log_file="logs.csv", save_every_n_steps=10):
        self.log_file = log_file
        self.save_every_n_steps = save_every_n_steps
        self.logs = []

        if os.path.exists(self.log_file):
            self.logs = pd.read_csv(self.log_file).to_dict(orient="records")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.logs.append(logs)

            if state.global_step % self.save_every_n_steps == 0:
                self.save_logs()

    def save_logs(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False)
        print(f"Logs saved to {self.log_file}")

    def get_dataframe(self):
        return pd.DataFrame(self.logs)


class DataAugmentationCallback(TrainerCallback):
    def __init__(self, trainer, full_dataset, prompt_formatter, num_epochs=5, step=800):
        self.trainer = trainer
        self.num_epochs = num_epochs
        self.dataset_generator = DataGenerator(self.trainer.model, self.trainer.tokenizer)
        self.ranges = [range(part, part + step) for part in range(2500, len(full_dataset), step)]
        self.current_range = 0
        self.full_dataset = full_dataset
        self.prompt_formatter = prompt_formatter

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % self.num_epochs == 0:
            new_samples = self.dataset_generator(self.full_dataset, range_gen=self.ranges[self.current_range])
            self.trainer.train_dataset = new_samples.map(self.prompt_formatter, batched=True)
