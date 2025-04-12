import fire
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, TrainerCallback
from trl import SFTTrainer

from src.callbacks import ClearMLCallback
from src.experiments.sft import SFTExperiment
from src.tools.model_builder import configured_build_model


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


class MOEExperiment(SFTExperiment):
    def prepare_model_and_tokenizer(self) -> (PreTrainedModel, PreTrainedTokenizer):
        self.train_args.model_max_length = self.cfg.model.max_length
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.name, padding_side="right", use_fast=True, trust_remote_code=True,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.cfg.model.special_tokens)})

        model = configured_build_model(self.cfg.model_args, self.train_args, update_tokenizer=tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer


def main(config: str):
    experiment = SFTExperiment(config)
    experiment.task_init()
    data_collator = experiment.prepare_datasets()

    eval_by_percent = {
        value: experiment.eval_dataset.filter(lambda example: example['percent'] == value)
        for value in [1]
    }

    clearml_callback = ClearMLCallback(experiment.task)
    multi_dataset_callback = MultiDatasetEvalCallback(eval_datasets=eval_by_percent, logger=clearml_callback.logger)

    trainer = SFTTrainer(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset.select(range(2)),  # Dummy eval dataset
        data_collator=data_collator,
        args=experiment.train_args,
        callbacks=[clearml_callback, multi_dataset_callback],
    )

    multi_dataset_callback.trainer = trainer
    trainer.add_callback(multi_dataset_callback)

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
