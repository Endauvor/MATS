import fire
from trl import DPOTrainer

from src.callbacks import ClearMLCallback
from src.experiments.dpo import UnslothExperiment


def main(config: str):
    experiment = UnslothExperiment(config)
    experiment.task_init()
    experiment.prepare_to_unsloth_dpo()

    trainer = DPOTrainer(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        train_dataset=experiment.train_dataset,
        args=experiment.train_args,
        callbacks=[ClearMLCallback(experiment.task)],
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
