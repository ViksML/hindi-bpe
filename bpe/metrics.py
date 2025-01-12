from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    iteration: int
    vocab_size: int
    tokens: int
    new_token: str
    frequency: int
    compression_ratio: float

class MetricsLogger:
    """Handles logging and saving of training metrics."""
    def __init__(self):
        self.token_logs: List[Dict] = []
        self.compression_logs: List[Dict] = []
        
    def log_iteration(self, metrics: TrainingMetrics):
        """Log metrics for current iteration."""
        self.token_logs.append({
            'iteration': metrics.iteration,
            'vocab_size': metrics.vocab_size,
            'tokens': metrics.tokens,
            'new_token': metrics.new_token,
            'frequency': metrics.frequency
        })
        
        self.compression_logs.append({
            'iteration': metrics.iteration,
            'compression_ratio': metrics.compression_ratio
        })
    
    def print_progress(self, metrics: TrainingMetrics, force: bool = False):
        """Print training progress."""
        if force or metrics.iteration % 500 == 0:
            print(f"\nIteration {metrics.iteration:,}:")
            print(f"Vocab size: {metrics.vocab_size:,}")
            print(f"Compression ratio: {metrics.compression_ratio:.2f}")
            print(f"New token: {metrics.new_token} (freq: {metrics.frequency:,})")
            print(f"Current tokens: {metrics.tokens:,}")
            print("-" * 50)
    
    def save(self, path: str):
        """Save metrics to file."""
        data = {
            'token_logs': self.token_logs,
            'compression_logs': self.compression_logs
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 