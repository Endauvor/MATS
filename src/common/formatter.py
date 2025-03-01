

def formatting_prompt(examples, tokenizer, max_length=1000):
    questions = examples["question"]
    answers = examples["answer"]
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for question, answer in zip(questions, answers):
        prompt = question
        full_text = prompt + tokenizer.bos_token + answer + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)
        tokenized_prompt = tokenizer(prompt, truncation=True, padding=False)

        prompt_length = len(tokenized_prompt["input_ids"])

        labels = tokenized_full["input_ids"].copy()

        labels[:prompt_length] = [-100] * prompt_length

        input_ids_list.append(tokenized_full["input_ids"])
        labels_list.append(labels)
        attention_mask_list.append(tokenized_full["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list  # Возвращаем attention_mask
    }
