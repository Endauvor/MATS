from typing import List

import torch
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from unsloth import FastLanguageModel


def process_answer(example):
    answer = example['answer']
    if "And .." in answer:
        split_index = answer.index("And ..")
        example['answer'] = answer[split_index:].strip()
    return example


class DataGenerator:
    def __init__(self, model, tokenizer, dataset_name="ExplosionNuclear/good_answers_chunk"):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

    def __call__(self, dataset: Dataset, range_gen=range(1500, 2500), **kwargs):
        FastLanguageModel.for_inference(self.model)
        new_dataset = self.generate_new_dataset(dataset, range_gen=range_gen, **kwargs)
        new_dataset.push_to_hub("_".join([self.dataset_name, str(range_gen.start), str(range_gen.stop)]))
        FastLanguageModel.for_training(self.model)

        return new_dataset

    def generate_answers(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        raw_answers = []
        for question, question_and_answer in zip(inputs["input_ids"], output):
            que_tokens = question.shape[0]
            raw_answers.append(question_and_answer[que_tokens:])

        raw_answers = self.tokenizer.batch_decode(raw_answers, skip_special_tokens=True)
        return list(map(lambda string: string.split("</simple_talk>")[0], raw_answers))

    def generate_new_dataset(
            self, train_data: Dataset, free_simtalk_size=0.8, range_gen=range(1500, 2500), batch_size=100
    ) -> Dataset:
        split_data = (train_data
                      .select(range_gen)
                      .train_test_split(test_size=free_simtalk_size))
        train_data_sq_fixed = split_data["train"]  # good simple talk in the answer
        train_data_sq_free = split_data["test"]  # we will generate simple talks
        answers = []

        for part_idx in tqdm(range(0, len(train_data_sq_free), batch_size)):
            right_border = min(len(train_data_sq_free), part_idx + batch_size)
            questions = train_data_sq_free.select(range(part_idx, right_border))['question']
            answers.extend(self.generate_answers(questions))

        premath_answers = []

        for idx, (qst, ans) in enumerate(zip(train_data_sq_free, answers)):
            premath_answers.append(
                {"question": qst["question"] + ans + '</simple_talk>'})  # now question contains generated simple talk

        new_data = Dataset.from_list(premath_answers)
        new_data = new_data.add_column("answer", train_data_sq_free["answer"])  # now there are questions and answers

        new_data = new_data.map(process_answer)  # delete initial simple talks
        return concatenate_datasets([
            new_data, train_data_sq_fixed.select_columns(["question", "answer"])
        ])
