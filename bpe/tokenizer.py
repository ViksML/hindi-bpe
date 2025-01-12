from typing import List, Set, Dict, Tuple
from collections import Counter
import re

class BaseTokenizer:
    """Base class for tokenizer implementations."""
    def __init__(self, vocab_size: int):
        if vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")
        self.vocab_size = vocab_size
        self.vocab: Set[str] = set()
    
    def tokenize(self, text: str) -> List[str]:
        """Convert text into list of tokens."""
        raise NotImplementedError
        
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        raise NotImplementedError

class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer."""
    def tokenize(self, text: str) -> List[str]:
        return list(text)
        
    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens) 