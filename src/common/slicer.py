import re

from src.common.templates import question_template, no_words_instruction, persent_instrunction


def get_qa(
        first_question, first_answer,
        second_question, second_answer,
        percent
):
    second_answer_previous, second_answer_cut = '', ''

    if percent == 0.01:
        # finding exact number
        match = re.search(r'\d+$', second_answer)
        if match:
            second_answer_cut = match.group()
            index = second_answer.find("####")
            second_answer_previous = second_answer[:index]

        question = question_template.format(
            first_question=first_question,
            second_question=second_question,
            persent_instrunction=no_words_instruction
        )
    else:
        # cutting
        start_idx = int(len(second_answer) * (1 - percent))

        second_answer_cut = adjust_substring(second_answer, start_idx)
        index = len(second_answer) - len(second_answer_cut)
        second_answer_previous = second_answer[:index]

        question = question_template.format(
            first_question=first_question,
            second_question=second_question,
            persent_instrunction=persent_instrunction.format(percent=percent * 100)
        )

    answer = f"""
    {first_answer} {second_answer_previous}

    And .. {second_answer_cut}!"""

    return {"question": question, "answer": answer, "percent": answer}


def adjust_substring(string, str_idx):
    """
    Returns a substring of P starting from an adjusted index.
    """
    if str_idx == 0:
        return string

    if str_idx < len(string) and string[str_idx - 1].isspace():
        return string[str_idx:]

    # Otherwise, we're in the middle of a word.
    i = str_idx
    # Move i forward until we find a whitespace (i.e. the end of the current word)
    while i < len(string) and not string[i].isspace():
        i += 1

    while i < len(string) and string[i].isspace():
        i += 1

    return string[i:]
