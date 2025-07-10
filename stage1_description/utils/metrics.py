import evaluate
import numpy as np
import torch
import gc

# Load metrics once outside
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        # eval_pred.predictions is now the prediction_ids processed by preprocess_logits_for_metrics.
        # From the returned tuple (predictions_ids, labels), we can get the predictions and labels.
        predictions_ids, labels = eval_pred.predictions[0], eval_pred.label_ids

        if np.any(predictions_ids < 0):
            predictions_ids[predictions_ids < 0] = tokenizer.pad_token_id

        # Decode the predicted token IDs to text
        decoded_preds = tokenizer.batch_decode(
            predictions_ids, skip_special_tokens=True)

        # Decode the labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        refs_for_bleu = [[label] for label in decoded_labels]

        # Calculate metrics
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
    This function runs on GPU. It converts logits to predicted IDs
    before they are gathered, helping to save memory.
    """

    # === DEBUGGING ===
    if logits is None:
        print("!!!!!! ERROR: `logits` IS NONE !!!!!!")
    # else:
    #     print(f"Type of `logits`: {type(logits)}")
    #     # If logits is a tuple, print the elements inside
    #     if isinstance(logits, tuple):
    #         print(f"  `logits` is a tuple with {len(logits)} elements.")
    #         for i, item in enumerate(logits):
    #             if hasattr(item, 'shape'):
    #                 print(f"  - Element {i}: type={type(item)}, shape={item.shape}")
    #             else:
    #                 print(f"  - Element {i}: type={type(item)}")
    #         # Assume logits actually lies in the first element
    #         logits = logits[0]
    #     else:
    #         print(f"`logits` shape: {logits.shape}")

    # Original logic
    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids, labels
