import argparse

from datasets import Dataset, load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams

from src.common.dataset import generate_data, update_answer, generate_data2

parser = argparse.ArgumentParser(description="Run dataset generation script.")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
parser.add_argument("--sqa_dataset_path", type=str, default="data/sQA_data", help="Path to dataset")
parser.add_argument("--model_name", type=str, default="casperhansen/llama-3.3-70b-instruct-awq", help="HF model link or local path")


def generate_dataset_bad1(inputs, questions_dataset, llm, sampling_params, batch_size):
    # Loop through inputs in batches (using a window of size batch_size)
    bad_answers_list = []

    for i in range(0, len(inputs), batch_size):
        batch_questions = inputs[i:i + batch_size]
        outputs = llm.generate(batch_questions, sampling_params=sampling_params)

        for k in range(batch_size):
            bad_answers_list.append({"bad_answer": outputs[k].outputs[0].text})

    bad_dataset = Dataset.from_list(bad_answers_list)
    bad_dataset = bad_dataset.add_column('percent', questions_dataset['percent'])
    return bad_dataset.add_column('question', questions_dataset['question2'])


def main(batch_size, sqa_dataset_path, model_name):
    llm = LLM(
        model=model_name,
        max_model_len=1000
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=300,
        top_p=0.9,
        stop="<\end>"
        # stop_words = ["Human:", "###", "\n\n"]
    )
    sqa = Dataset.load_from_disk("data/sQA_data")
    gsm8k = load_dataset('Openai/gsm8k', 'main')

    simpletalks = sqa.map(update_answer, batched=True)
    questions_dataset = Dataset.from_generator(generate_data(simpletalks, gsm8k))

    bad_dataset = generate_dataset_bad1(
        questions_dataset["question"], questions_dataset, llm, sampling_params, args.batch_size
    )
    bad_dataset.save_to_disk("data/dpo_dataset/bad1")

    # bad_dataset = Dataset.load_from_disk("data/dpo_dataset/bad1")

    bad_dataset2 = Dataset.from_generator(generate_data2(simpletalks, gsm8k))
    # bad_dataset_final = concatenate_datasets([bad_dataset, bad_dataset2])
    bad_dataset2.save_to_disk("data/dpo_dataset/bad2")


if __name__ == "__main__":

    args = parser.parse_args()
    main(**vars(args))



