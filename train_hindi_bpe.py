from bpe import HindiBPE
from bpe.visualization import BPEVisualizer
import os

def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        os.path.join('models', 'hindi_bpe'),
        os.path.join('stats', 'hindi_bpe'),
        os.path.join('stats', 'hindi_bpe', 'plots'),
        os.path.join('data', 'hindi'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Create directories if they don't exist
MODEL_DIR = os.path.join('models', 'hindi_bpe')
STATS_DIR = os.path.join('stats', 'hindi_bpe')
DATA_DIR = os.path.join('data', 'hindi')
create_directory_structure()

def main():
    # Load Hindi text data
    with open(os.path.join(DATA_DIR, 'text.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
        print(f"\nLoaded {len(text):,} characters of text")
        print(f"Unique characters: {len(set(text)):,}")
    
    # Initialize and train BPE
    bpe = HindiBPE(vocab_size=5000)
    print("\nStarting BPE training...")
    bpe.fit(text)
    
    # Save the model and metrics
    model_path = os.path.join(MODEL_DIR, 'model.json')
    stats_path = os.path.join(STATS_DIR, 'metrics.json')
    bpe.save(model_path, stats_path)
    
    # Generate and save visualization plots
    visualizer = BPEVisualizer(STATS_DIR)
    visualizer.plot_training_stats(bpe.metrics)
    print("\nGenerated visualization plots in:", os.path.join(STATS_DIR, 'plots'))
    
    # Test encoding
    test_text = "आप कैसे हैं?"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"\nTest encoding/decoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()