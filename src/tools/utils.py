import re
from datasets import Dataset

reasoning_start = "<simple_talk>"
reasoning_end   = "</simple_talk>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}(.+?){reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

match_numbers = re.compile(
    rf"{solution_start}\s*([+-]?\d+(?:\.\d+)?)",
    flags = re.MULTILINE | re.DOTALL
)

match_simpletalk = re.compile(
    rf"{reasoning_start}(.+?){reasoning_end}",
    flags=re.DOTALL
)

def extract_task_and_answer(example):
   
    instruction = example.get('instruction', '')
    full_answer = example.get('full_answer', '') 

    task_description = None
    
    instruction_match = re.search(
        r"Here's your specific task:\s*(.*?)\s*Do not reference", # Упрощенный шаблон
        instruction,
        re.DOTALL | re.IGNORECASE
    )

    if instruction_match:
        task_description = instruction_match.group(1)
        task_description = task_description.strip()
        if task_description.endswith('.'):
            task_description = task_description[:-1].strip()

    final_answer = None
    answer_match = re.search(r"The answer:\s*(-?\d+)\s*!", full_answer, re.IGNORECASE)
    if answer_match:
        final_answer = answer_match.group(1)


    return {
        'task_description': task_description[3:],
        'final_answer': final_answer
    }


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(solution_start)  == 1 else -0.5
        score += 0.5 if response.count(solution_end)    == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(2)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close 
            try:
                diff = abs(float(guess) - float(true_answer)) / true_answer
                if   diff < 0.05: score += 1.0
                if   diff < 0.1: score += 0.5
                elif diff < 0.15: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # print('*'*20, f"Question:\n{question}", f"\nCorrect Math Answer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess       = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores

def check_simpletalk(completions, **kwargs):
    
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_simpletalk.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    
    for guess in extracted_responses:
        if guess is None:
            scores.append(0)
            continue

        digits = re.findall(r"\d", guess)
        count = len(digits)

        if count > 10:
            scores.append(-5)
        elif count > 5:
            scores.append(-3)
        elif count > 2:
            scores.append(-1)
        elif count > 0:
            scores.append(-0.5)
        else:
            scores.append(2)
        
    return scores

    