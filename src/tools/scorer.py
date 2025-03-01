from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.tools.plot import plot_for_sample


class Scorer:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, dataset_samples, batch_size=20):

        all_results = []
        for batch_index in tqdm(range(0, len(dataset_samples), batch_size)):
            batch_samples = dataset_samples[batch_index:batch_index + batch_size]

            questions = batch_samples['question']
            answers = batch_samples['answer']

            batch_results = self.compute_token_metrics(questions, answers, batch_index)
            all_results.extend(batch_results)

        return pd.DataFrame(all_results)

    def compute_token_metrics(self, questions: List[str], answers: List[str], batch_index=0) -> List[dict]:
        """
        Compute entropy and cross-entropy loss for each token
        Args:
            questions: list of questions
            answers: list of answers
            batch_index: index of the batch

        Returns: structures with batch_sample, token_position, token, loss, entropy
        """
        prompts = [q.strip() + " " + a.strip() for q, a in zip(questions, answers)]

        # Tokenize prompts
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch, seq_len, vocab_size)

        loss_per_token, entropy, shift_labels = self.calculate_loss_and_entropy(logits, input_ids)
        batch_size, seq_len = shift_labels.shape

        question_lengths = []
        for q in questions:
            q_enc = self.tokenizer(q.strip(), return_tensors="pt")
            question_lengths.append(q_enc.input_ids.shape[1])

        loss_per_token_np = loss_per_token.float().cpu().numpy()
        entropy_np = entropy.float().cpu().numpy()
        input_ids_np = input_ids.cpu().numpy()

        results = []

        for i in range(batch_size):
            # Decode original sequence
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids_np[i])
            q_len = question_lengths[i]

            ppl = self.calculate_ppl(loss_per_token_np[i, q_len - 1:])

            for j in range(q_len, len(tokens)):
                shift_index = j - 1  # return indexes of the shifted tokens
                if shift_index < loss_per_token_np.shape[1]:
                    results.append({
                        "batch_sample": batch_index + i,
                        "token_position": j,
                        "token": tokens[j],
                        "loss": loss_per_token_np[i, shift_index],
                        "entropy": entropy_np[i, shift_index],
                        "ppl": ppl
                    })
        return results

    def calculate_ppl(self, answer_loss):
        if len(answer_loss) > 0:
            return np.exp(np.mean(answer_loss))
        else:
            return np.nan

    def calculate_loss_and_entropy(self, logits, input_ids):

        # Shift all tokens on the right by one position
        shift_logits = logits[:, :-1, :]  # shape: (batch, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:]  # shape: (batch, seq_len-1)

        loss_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        ).view(shift_labels.size())  # shape: (batch, seq_len-1)

        # Entropy: H = -sum(P * log P)
        probs = torch.softmax(shift_logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # shape: (batch, seq_len-1)

        return loss_per_token, entropy, shift_labels

    def plot_sample(self, df_sample):
        plot_for_sample(df_sample)
