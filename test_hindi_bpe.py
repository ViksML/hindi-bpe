import unittest
from bpe import HindiBPE, TrainingMetrics
import json
import os
import tempfile
import shutil

# Use the same directory structure as train_hindi_bpe.py
MODEL_DIR = os.path.join('models', 'hindi_bpe')
STATS_DIR = os.path.join('stats', 'hindi_bpe')
DATA_DIR = os.path.join('data', 'hindi')

class TestHindiBPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create necessary directories before any tests."""
        for directory in [MODEL_DIR, STATS_DIR, DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

    def setUp(self):
        """Set up test data."""
        self.test_text = """
        नमस्ते भारत। यह एक परीक्षण वाक्य है।
        हिंदी भाषा बहुत सुंदर है।
        मैं हिंदी सीख रहा हूं।
        """
        self.bpe = HindiBPE(vocab_size=100)
        
    def test_initialization(self):
        """Test BPE initialization."""
        self.assertEqual(self.bpe.vocab_size, 100)
        self.assertEqual(len(self.bpe.merges), 0)
        self.assertEqual(len(self.bpe.vocab), 0)
        
    def test_get_stats(self):
        """Test pair frequency counting."""
        words = [['न', 'म', 'स्', 'ते']]
        stats = self.bpe.get_stats(words)
        self.assertGreater(len(stats), 0)
        self.assertIsInstance(stats[('न', 'म')], int)
        
    def test_merge_vocab(self):
        """Test merging tokens."""
        words = [['न', 'म', 'स्', 'ते']]
        pair = ('न', 'म')
        new_token = 'नम'
        merged = self.bpe.merge_vocab(words, pair, new_token)
        self.assertEqual(merged[0][0], 'नम')
        
    def test_fit(self):
        """Test BPE training."""
        self.bpe.fit(self.test_text)
        
        # Check if vocabulary was created
        self.assertGreater(len(self.bpe.vocab), 0)
        self.assertLess(len(self.bpe.vocab), self.bpe.vocab_size)
        
        # Check if merges were learned
        self.assertGreater(len(self.bpe.merges), 0)
        
        # Check if logs were created
        self.assertGreater(len(self.bpe.metrics.token_logs), 0)
        self.assertGreater(len(self.bpe.metrics.compression_logs), 0)
        
    def test_encode(self):
        """Test text encoding."""
        self.bpe.fit(self.test_text)
        test_word = "नमस्ते"
        encoded = self.bpe.encode(test_word)
        
        # Check if encoding produces tokens
        self.assertGreater(len(encoded), 0)
        
        # Check if all tokens are in vocabulary
        for token in encoded:
            self.assertIn(token, self.bpe.vocab)
            
    def test_save_load(self):
        """Test model saving and loading."""
        # Train the model
        self.bpe.fit(self.test_text)
        
        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_model_dir = os.path.join(tmp_dir, 'models', 'hindi_bpe')
            test_stats_dir = os.path.join(tmp_dir, 'stats', 'hindi_bpe')
            os.makedirs(test_model_dir)
            os.makedirs(test_stats_dir)
            
            model_path = os.path.join(test_model_dir, 'model.json')
            stats_path = os.path.join(test_stats_dir, 'metrics.json')
            
            try:
                # Save the model
                self.bpe.save(model_path, stats_path)
                
                # Load into new instance
                new_bpe = HindiBPE(vocab_size=100)
                new_bpe.load(model_path, stats_path)
                
                # Compare attributes
                self.assertEqual(self.bpe.vocab, new_bpe.vocab)
                self.assertEqual(len(self.bpe.merges), len(new_bpe.merges))
                self.assertEqual(len(self.bpe.metrics.token_logs), len(new_bpe.metrics.token_logs))
                self.assertEqual(len(self.bpe.metrics.compression_logs), len(new_bpe.metrics.compression_logs))
                
                # Compare encodings
                test_word = "नमस्ते"
                self.assertEqual(
                    self.bpe.encode(test_word),
                    new_bpe.encode(test_word)
                )
                
            finally:
                # Clean up is handled by TemporaryDirectory
                pass
            
    def test_compression_ratio(self):
        """Test if compression ratio is calculated correctly."""
        self.bpe.fit(self.test_text)
        
        # Get final compression ratio
        final_compression = self.bpe.metrics.compression_logs[-1]['compression_ratio']
        
        # Check if compression ratio is positive
        self.assertGreater(final_compression, 0)
        
        # Check if compression ratios are monotonically increasing
        ratios = [log['compression_ratio'] for log in self.bpe.metrics.compression_logs]
        for i in range(1, len(ratios)):
            self.assertGreaterEqual(ratios[i], ratios[i-1])
            
    def test_error_handling(self):
        """Test error handling."""
        # Test with empty text
        with self.assertRaises(Exception):
            self.bpe.fit("")
            
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.bpe.load("non_existent_file.json")
            
        # Test with invalid vocabulary size
        with self.assertRaises(Exception):
            HindiBPE(vocab_size=0)
            
    def test_existing_model_token_count(self):
        """Test that the existing BPE model has appropriate token count."""
        try:
            # Load the existing model
            bpe = HindiBPE(vocab_size=5000)
            model_path = os.path.join(MODEL_DIR, 'model.json')
            stats_path = os.path.join(STATS_DIR, 'metrics.json')
            bpe.load(model_path, stats_path)
            
            # Get total vocabulary size
            total_tokens = len(bpe.vocab)
            
            # Check token count
            self.assertGreaterEqual(
                total_tokens,
                4500,  # Adjusted minimum token requirement
                f"Model should have at least 4500 tokens, but found only {total_tokens}"
            )
            
            # Load some test text to verify tokens are usable
            try:
                with open(os.path.join(DATA_DIR, 'text.txt'), 'r', encoding='utf-8') as f:
                    test_text = f.read()[:1000]  # Use first 1000 chars for testing
            except FileNotFoundError:
                test_text = "नमस्ते भारत। यह एक परीक्षण वाक्य है।"
            
            # Encode test text
            encoded = bpe.encode(test_text)
            unique_tokens_used = len(set(encoded))
            
            # Verify tokens are actually being used
            self.assertGreater(
                unique_tokens_used,
                100,  # Conservative lower bound for tokens in use
                f"Expected more than 100 unique tokens in use, but got {unique_tokens_used}"
            )
            
            # Check compression ratio from logs
            final_compression = bpe.metrics.compression_logs[-1]['compression_ratio']
            self.assertGreaterEqual(
                final_compression,
                3.0,
                f"Expected compression ratio >= 3.0, but got {final_compression}"
            )
            
        except FileNotFoundError:
            self.skipTest(f"Model not found at {model_path}. Run train_hindi_bpe.py first.")

    def test_metrics_logging(self):
        """Test that metrics are properly logged."""
        self.bpe.fit(self.test_text)
        
        # Check token logs
        self.assertTrue(len(self.bpe.metrics.token_logs) > 0)
        first_log = self.bpe.metrics.token_logs[0]
        self.assertIn('iteration', first_log)
        self.assertIn('vocab_size', first_log)
        self.assertIn('tokens', first_log)
        self.assertIn('new_token', first_log)
        self.assertIn('frequency', first_log)
        
        # Check compression logs
        self.assertTrue(len(self.bpe.metrics.compression_logs) > 0)
        first_compression = self.bpe.metrics.compression_logs[0]
        self.assertIn('iteration', first_compression)
        self.assertIn('compression_ratio', first_compression)
        
    def test_decode(self):
        """Test token decoding."""
        self.bpe.fit(self.test_text)
        test_word = "नमस्ते"
        encoded = self.bpe.encode(test_word)
        decoded = self.bpe.decode(encoded)
        # Remove spaces from decoded text for comparison
        decoded_clean = ''.join(decoded.split())
        self.assertEqual(test_word, decoded_clean)

if __name__ == '__main__':
    unittest.main(verbosity=2) 