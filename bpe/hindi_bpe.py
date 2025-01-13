from typing import List, Tuple, Dict, Set
from collections import Counter
import json
import re
from .tokenizer import BaseTokenizer
from .metrics import TrainingMetrics, MetricsLogger
import os

class HindiBPE(BaseTokenizer):
    """Byte-Pair Encoding implementation for Hindi text."""
    
    def __init__(self, vocab_size: int = 5000):
        super().__init__(vocab_size)
        self.merges: Dict[Tuple[str, str], str] = {}
        self.metrics = MetricsLogger()
        self.min_freq_threshold = 3  # Reduced from 5 to handle more diverse text
        
    def get_stats(self, words: List[List[str]]) -> Counter:
        """Count pair frequencies in current vocabulary."""
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i + 2])] += 1
        return pairs

    def merge_vocab(self, words: List[List[str]], pair: Tuple[str, str], new_token: str) -> List[List[str]]:
        """Merge all occurrences of a pair into a new token."""
        new_words = []
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in words:
            w = ' '.join(word)
            w = pattern.sub(new_token, w)
            new_words.append(w.split())
            
        return new_words

    def fit(self, text: str, min_freq: int = 2):
        """Train BPE on input text."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Initialize with characters
        words = [[c for c in word] for word in text.split()]
        self.vocab = set(char for word in words for char in word)
        
        original_tokens = sum(len(word) for word in words)
        
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(words)
            if not pairs:
                break
                
            most_common = pairs.most_common(1)[0]
            if most_common[1] < min_freq:
                break
                
            pair, count = most_common
            new_token = ''.join(pair)
            self.merges[pair] = new_token
            self.vocab.add(new_token)
            
            words = self.merge_vocab(words, pair, new_token)
            
            # Calculate metrics
            current_tokens = sum(len(word) for word in words)
            metrics = TrainingMetrics(
                iteration=iteration,
                vocab_size=len(self.vocab),
                tokens=current_tokens,
                new_token=new_token,
                frequency=count,
                compression_ratio=original_tokens / current_tokens
            )
            
            # Log metrics
            self.metrics.log_iteration(metrics)
            self.metrics.print_progress(metrics)
            
            iteration += 1
        
        # Print final statistics
        final_metrics = TrainingMetrics(
            iteration=iteration,
            vocab_size=len(self.vocab),
            tokens=current_tokens,
            new_token=new_token,
            frequency=count,
            compression_ratio=original_tokens / current_tokens
        )
        self.metrics.print_progress(final_metrics, force=True)

    def encode(self, text: str) -> List[str]:
        """Encode text using learned BPE merges."""
        words = [[c for c in word] for word in text.split()]
        for pair, new_token in self.merges.items():
            words = self.merge_vocab(words, pair, new_token)
        return [token for word in words for token in word]
        
    def decode(self, tokens: List[str]) -> str:
        """Decode tokens back to text."""
        return ' '.join(''.join(tokens))

    def save(self, model_path: str, stats_path: str = None):
        """Save BPE model to file."""
        data = {
            'merges': {' '.join(k): v for k, v in self.merges.items()},
            'vocab': list(self.vocab)
        }
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # Save metrics separately
        if stats_path:
            self.metrics.save(stats_path)

    def load(self, model_path: str, stats_path: str = None):
        """Load BPE model from file."""
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
        self.vocab = set(data['vocab'])
        
        # Load metrics if available
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
                self.metrics.token_logs = metrics_data['token_logs']
                self.metrics.compression_logs = metrics_data['compression_logs']
        
  