import fire
from trl import SFTTrainer

from src.callbacks import ClearMLCallback
from src.experiments.sft import SFTExperiment


def main(config: str):
    experiment = SFTExperiment(config)
    experiment.setup_lora()
    experiment.task_init()
    data_collator = experiment.prepare_datasets()

    trainer = SFTTrainer(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset,
        data_collator=data_collator,
        args=experiment.train_args,
        callbacks=[ClearMLCallback(experiment.task)],
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
