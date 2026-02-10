import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from collections import Counter
import os
import re
from evaluate import load

# NLTK tokenization has been replaced with simple regex-based tokenization
# to avoid dependency on NLTK data downloads

# Load evaluation metrics from Hugging Face Evaluate.
#bleu_metric = load("bleu")
# Lazy-load ROUGE to avoid module-level downloads in each worker process
rouge_metric = None
#meteor_metric = load("meteor")
#bertscore_metric = load("bertscore")


class TextSimilarity:

    def __init__(self, method='jaccard'):
        available_methods = [
            'jaccard', 'dice', 'tfidf_cosine', 'overlap', 'bleu', 'hamming', 'rouge', 'meteor', 'bertscore'
        ]
        if method not in available_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {available_methods}")
        self.method = method

    def compute_score(self, solution_str, ground_truth, extra_info):
        method_func = getattr(self, f'_{self.method}')
        target_score = method_func(solution_str, ground_truth)
        prefix = extra_info.get('question', '')
        prefix_score = method_func(solution_str, prefix)
        # Combine scores: target_score penalized by prefix_score
        # Option 1: Simple multiplication (if prefix_score is high, it reduces target_score less)
        #combined_score = target_score * (1 - 0.5 * prefix_score)
        
        # Option 2: Weighted average favoring target_score
        #combined_score = 0.6 * target_score + 0.4 * (1 - prefix_score)
        
        # Option 3: Exponential penalty (more aggressive penalty for high prefix_score)
        # combined_score = target_score * (1 - prefix_score ** 2)
        # 10/04/2025: we use target_score only to assess whether generation matches ground truth more
        # this correspond to run2 in finemath_rm_rl
        return target_score
    
    def _tokenize(self, text):
        """Tokenize text into a set of lowercase word tokens."""
        # Use simple split tokenization to avoid NLTK downloads
        # Simple tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    def _tokenize_list(self, text):
        """Tokenize text into a list of lowercase word tokens."""
        # Use simple split tokenization to avoid NLTK downloads
        # Simple tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _jaccard(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0

    def _dice(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) else 0.0

    def _overlap(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        return intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) else 0.0

    def _tfidf_cosine(self, text1, text2):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2]).toarray()
        similarity = 1 - cosine(vectors[0], vectors[1])
        return float(similarity) if not np.isnan(similarity) else 0.0

    def _bleu(self, text1, text2):
        """
        For BLEU, the candidate is text1 and the reference is text2.
        The evaluate BLEU metric expects references as a nested list.
        """
        predictions = [text1]
        references = [[text2]]  # Extra list layer indicating one or more references per candidate.
        result = bleu_metric.compute(predictions=predictions, references=references)
        return float(result['bleu'])

    def _hamming(self, text1, text2):
        # Convert strings to lists of characters.
        s1 = list(text1.lower())
        s2 = list(text2.lower())
        # If strings have different lengths, pad the shorter one with spaces.
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0  # Both strings are empty.
        s1.extend([' '] * (max_len - len(s1)))
        s2.extend([' '] * (max_len - len(s2)))
        # Calculate Hamming distance.
        distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        # Normalize so that 1 means identical.
        return 1.0 - (distance / max_len)

    def _rouge(self, text1, text2):
        """
        Compute ROUGE score. Here we use the F1 (F-measure) of ROUGE-L.
        The evaluate metric now returns a numeric value for 'rougeL'.
        """
        global rouge_metric
        if rouge_metric is None:
            # Lazy initialization so only processes that actually use ROUGE will load it
            rouge_metric = load("rouge")
        predictions = [text1]
        references = [text2]
        result = rouge_metric.compute(predictions=predictions, references=references)
        # Directly extract the numeric value for ROUGE-L F1 score.
        score = result.get("rougeL", 0.0)
        return float(score)

    def _meteor(self, text1, text2):
        """
        Compute METEOR score using the evaluate metric.
        METEOR typically returns a value between 0 and 1.
        """
        predictions = [text1]
        references = [text2]  # METEOR expects references as a list of strings.
        result = meteor_metric.compute(predictions=predictions, references=references)
        return float(result['meteor'])

    def _bertscore(self, text1, text2):
        """
        Compute BERTScore F1 for the candidate and reference.
        The compute method returns a dictionary with lists for precision, recall, and f1.
        We return the first score.
        """
        predictions = [text1]
        references = [text2]
        result = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
        # result['f1'] is a list of scores (one for each input pair).
        return float(result['f1'][0])
