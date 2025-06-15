from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from src.utils.schema import MetricSchema

class MetricEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        self.smooth_fn = SmoothingFunction().method1 # Prevent BLEU=0 for short texts

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute average ROUGE-1 and ROUGE-2 F1 scores.

        Args:
            predictions: List of generated texts.
            references: List of ground truth texts.

        Returns:
            Dict with 'rouge1' and 'rouge2' average F1 scores.
        """
        scores = {"rouge1": [], "rouge2": []}

        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores["rouge1"].append(score["rouge1"].fmeasure)
            scores["rouge2"].append(score["rouge2"].fmeasure)

        return {
            MetricSchema.ROUGE1: sum(scores["rouge1"]) / len(scores["rouge1"]),
            MetricSchema.ROUGE2: sum(scores["rouge2"]) / len(scores["rouge2"])
        }

    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute average BLEU-1 and BLEU-2 scores.

        Args:
            predictions: List of generated texts.
            references: List of ground truth texts.

        Returns:
            Dict with 'bleu1' and 'bleu2' average scores.
        """
        scores = {"bleu1": [], "bleu2": []}

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]

            if not pred_tokens or not ref_tokens[0]:
                scores["bleu1"].append(0.0)
                scores["bleu2"].append(0.0)
                continue

            scores["bleu1"].append(
                sentence_bleu(ref_tokens, pred_tokens, weights=(1.0, 0.0), smoothing_function=self.smooth_fn)
            )
            scores["bleu2"].append(
                sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5), smoothing_function=self.smooth_fn)
            )

        return {
            MetricSchema.BLEU1: sum(scores["bleu1"]) / len(scores["bleu1"]),
            MetricSchema.BLEU2: sum(scores["bleu2"]) / len(scores["bleu2"])
        }