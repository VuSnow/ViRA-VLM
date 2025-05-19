from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import torch


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def calculate_bleu_scores(references_lists, hypotheses):
    chencherry = SmoothingFunction()
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}

    if not hypotheses or not references_lists:
        return {k: 0 for k in bleu_scores.keys()}

    for refs_str_list, hyp_str in zip(references_lists, hypotheses):
        if not hyp_str:  # Bỏ qua nếu giả thuyết rỗng
            bleu_scores["BLEU-1"].append(0)
            bleu_scores["BLEU-2"].append(0)
            bleu_scores["BLEU-3"].append(0)
            bleu_scores["BLEU-4"].append(0)
            continue

        tokenized_refs = [nltk.word_tokenize(
            ref.lower()) for ref in refs_str_list]
        tokenized_hyp = nltk.word_tokenize(hyp_str.lower())

        # Kiểm tra tokenized_hyp rỗng để tránh lỗi division by zero trong sentence_bleu
        if not tokenized_hyp:
            bleu_scores["BLEU-1"].append(0)
            bleu_scores["BLEU-2"].append(0)
            bleu_scores["BLEU-3"].append(0)
            bleu_scores["BLEU-4"].append(0)
            continue

        bleu_scores["BLEU-1"].append(sentence_bleu(tokenized_refs, tokenized_hyp,
                                     weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
        bleu_scores["BLEU-2"].append(sentence_bleu(tokenized_refs, tokenized_hyp, weights=(
            0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
        bleu_scores["BLEU-3"].append(sentence_bleu(tokenized_refs, tokenized_hyp, weights=(
            0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1))
        bleu_scores["BLEU-4"].append(sentence_bleu(tokenized_refs, tokenized_hyp, weights=(
            0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))

    avg_bleu = {k: sum(v) / len(v) if v else 0 for k, v in bleu_scores.items()}
    return avg_bleu

# Hàm tính điểm ROUGE


def calculate_rouge_scores(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_agg = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    if not hypotheses or not references:
        return {"ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0}

    for ref_str, hyp_str in zip(references, hypotheses):
        if not hyp_str:  # Bỏ qua nếu giả thuyết rỗng
            rouge_scores_agg['rouge1'].append(0)
            rouge_scores_agg['rouge2'].append(0)
            rouge_scores_agg['rougeL'].append(0)
            continue
        scores = scorer.score(ref_str, hyp_str)
        rouge_scores_agg['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores_agg['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores_agg['rougeL'].append(scores['rougeL'].fmeasure)

    avg_rouge = {k: sum(v) / len(v) if v else 0 for k,
                 v in rouge_scores_agg.items()}
    return {
        "ROUGE-1": avg_rouge.get('rouge1', 0),
        "ROUGE-2": avg_rouge.get('rouge2', 0),
        "ROUGE-L": avg_rouge.get('rougeL', 0)
    }


def calculate_token_accuracy(logits, labels, pad_token_id):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len]
    Trả về accuracy ignoring padding (-100 or pad_token_id)

    Lấy pred token = argmax logits mỗi vị trí, so sánh với label.
    """
    preds = torch.argmax(logits, dim=-1)
    mask = (labels != -100) & (labels != pad_token_id)

    correct = (preds == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    return accuracy
