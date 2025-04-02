import argparse

from dotenv import load_dotenv
from trl import DPOTrainer

from src.callbacks import ClearMLCallback
from src.containers.dpo import UnslothExperiment

load_dotenv()


parser = argparse.ArgumentParser(description="Run DPO training")
parser.add_argument(
    "-conf", "--config", default="configs/dpo/qwen2_5_7b.yaml", type=str, help="Path to config file"
)

if __name__ == "__main__":
    args = parser.parse_args()
    experiment = UnslothExperiment(args.config)
    experiment.task_init()
    experiment.prepare_to_unsloth_dpo()

    trainer = DPOTrainer(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        train_dataset=experiment.train_dataset,
        args=experiment.train_args,
        callbacks=[
            ClearMLCallback(experiment.task)
        ],
    )

    trainer.train()
