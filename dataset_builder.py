import argparse

from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="Run dataset generation script.")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
parser.add_argument("--sqa_dataset_path", type=str, default="data/sQA_data", help="Path to dataset")
parser.add_argument("--model_name", type=str, default="casperhansen/llama-3.3-70b-instruct-awq", help="HF model link or local path")


def generate_full_dataset(full_dataset, llm, sampling_params, batch_size):
    # Loop through inputs in batches (using a window of size batch_size)
    bad_answers_list = []

    prompts = full_dataset['queries_for_b_answers']
    for i in range(0, len(prompts), batch_size):
        batch_questions = prompts[i:i + batch_size]
        outputs = llm.generate(batch_questions, sampling_params=sampling_params)

        for k in range(batch_size):
            bad_answers_list.append({"bad_answer": outputs[k].outputs[0].text})

    bad_dataset = Dataset.from_list(bad_answers_list)
    return full_dataset.add_column('bad_answer', bad_dataset['bad_answer'])


def main(batch_size, model_name, **kwargs):
    llm = LLM(
        model=model_name,
        max_model_len=1000
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=300,
        top_p=0.9,
        stop="<\end>"
    )

    full_dataset = load_dataset("ExplosionNuclear/good_answers")['train']

    full_dataset = generate_full_dataset(full_dataset, llm, sampling_params, batch_size)
    full_dataset.save_to_disk("data/dpo_dataset/full")
    full_dataset.rename_columns({"question": "prompt", "good_answer": "chosen", "bad_answer": "rejected"})
    full_dataset.save_to_disk("data/dpo_dataset/full")
    full_dataset.push_to_hub("ExplosionNuclear/dpo_dataset_final")


if __name__ == "__main__":

    args = parser.parse_args()
    main(**vars(args))



