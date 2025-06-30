import evaluate
import numpy as np
import torch
import gc

# Tải các metrics một lần bên ngoài
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        # eval_pred.predictions bây giờ là các prediction_ids đã được xử lý trước
        # nhờ vào preprocess_logits_for_metrics.
        predictions_ids, labels = eval_pred.predictions, eval_pred.label_ids

        if np.any(predictions_ids < 0):
            predictions_ids[predictions_ids < 0] = tokenizer.pad_token_id

        # Decode các token IDs dự đoán thành văn bản
        decoded_preds = tokenizer.batch_decode(
            predictions_ids, skip_special_tokens=True)

        # Decode các labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Xử lý hậu kỳ
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        refs_for_bleu = [[label] for label in decoded_labels]

        # Tính toán metrics
        bleu_score = bleu.compute(
            predictions=decoded_preds, references=refs_for_bleu)
        rouge_score = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        meteor_score = meteor.compute(
            predictions=decoded_preds, references=decoded_labels)

        result = {
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rougeL": rouge_score["rougeL"],
            "meteor": meteor_score["meteor"]
        }

        return result

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """
    Hàm này chạy trên GPU. Nó chuyển đổi logits thành các ID dự đoán
    trước khi chúng được gom lại, giúp tiết kiệm bộ nhớ.
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    pred_ids = torch.argmax(logits, dim=-1)
    pred_ids = pred_ids.to(dtype=torch.int32)
    return pred_ids
