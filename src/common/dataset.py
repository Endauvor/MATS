import random
import re

from src.common.slicer import get_qa
from src.common.templates import question_for_b_answer_template, question_template, no_words_instruction, \
    persent_instrunction


def remove_strange_symbols(data):
    """
    Removes all occurrences of "<< ... >>" from the text in data['answer'].
    """
    answer = data.get('answer')
    if isinstance(answer, str):
        pattern = r"<<.*?>>"
        cleaned_answer = re.sub(pattern, "", answer)
        data['answer'] = cleaned_answer
    return data


def cut_the_talk(text, n):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    first_n_sentences = " ".join(sentences[:n])
    return first_n_sentences


def update_answer(
        examples,
        weights=[0.1, 0.3, 0.3, 0.2, 0.1]  # for 1, 2, 3 and 4 sentences respectively
):
    for idx, talk in enumerate(examples['answer']):
        n = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
        examples['answer'][idx] = cut_the_talk(talk, n)
    return examples


def get_q_for_gen_banswer(
        simple_talk, first_question,
        math_question, math_answer,
        percent  # style
):
    question = question_for_b_answer_template.format(
        simple_talk=simple_talk,
        math_answer=math_answer,
        math_question=math_question,
        percent1=round(percent * 100),
        percent2=round((1 - percent) * 100)
        # style=style)
    )

    if round(1 - percent, 2) == 0.01:

        question2 = question_template.format(
            first_question=first_question,
            second_question=math_question,
            persent_instrunction=no_words_instruction
        )

    else:

        question2 = question_template.format(
            first_question=first_question,
            second_question=math_question,
            persent_instrunction=persent_instrunction.format(percent=round((1 - percent) * 100))
        )

    return {"question2": question2, "question": question, "percent": round((1 - percent) * 100)}


def generate_data(
        simpletalks, gmsk8,
        percentage=[0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1],
        num_mixin=6000  # the number of creative mixings (remeaining with simply insertion is 1400)
):
    def generator():
        simple_talk_chunk = simpletalks.select(range(num_mixin))
        math_chunk = gmsk8["train"].select(range(num_mixin))
        for idx, (sqa, gmsk) in enumerate(zip(simple_talk_chunk, math_chunk)):
            percent = random.choices(percentage, weights=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])[0]

            yield get_q_for_gen_banswer(
                sqa["answer"], sqa["question"],
                gmsk["question"], gmsk["answer"],
                percent
            )

    return generator


def generate_data2(
        simpletalks, gmsk8,
        percentage=[0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
        num_mixin=6000
):

    weights = [0.1] * (len(percentage) - 1) + [0.3]

    def generator():
        simple_talk_chunk = simpletalks.select(range(num_mixin, len(gmsk8["train"])))
        math_chunk = gmsk8["train"].select(range(num_mixin, len(gmsk8["train"])))

        for idx, (sqa, gmsk) in enumerate(zip(simple_talk_chunk, math_chunk)):
            percent = random.choices(percentage, weights=weights)[0]
            yield get_qa(
                sqa["question"], sqa["answer"],
                gmsk["question"], gmsk["answer"],
                percent
            )

    return generator
