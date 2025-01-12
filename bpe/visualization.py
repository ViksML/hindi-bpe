import matplotlib.pyplot as plt
import os

class BPEVisualizer:
    """Visualizes BPE training statistics."""
    
    def __init__(self, stats_dir: str):
        self.stats_dir = stats_dir
        self.plots_dir = os.path.join(stats_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_training_stats(self, metrics_logger):
        """Generate all training statistics plots."""
        # Get data from metrics
        iterations = [log['iteration'] for log in metrics_logger.token_logs]
        vocab_sizes = [log['vocab_size'] for log in metrics_logger.token_logs]
        token_freqs = [log['frequency'] for log in metrics_logger.token_logs]
        compression_ratios = [log['compression_ratio'] for log in metrics_logger.compression_logs]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Vocabulary Size Growth
        ax1 = fig.add_subplot(221)
        ax1.plot(iterations, vocab_sizes)
        ax1.set_title('Vocabulary Size Growth')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Vocabulary Size')
        ax1.grid(True)
        
        # 2. Compression Ratio Progress
        ax2 = fig.add_subplot(222)
        ax2.plot(iterations, compression_ratios)
        ax2.set_title('Compression Ratio Progress')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Compression Ratio')
        ax2.grid(True)
        
        # 3. Token Frequencies (Log Scale)
        ax3 = fig.add_subplot(223)
        ax3.plot(iterations, token_freqs)
        ax3.set_title('Token Frequencies')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Frequency')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 4. Compression Ratio Distribution
        ax4 = fig.add_subplot(224)
        ax4.hist(compression_ratios, bins=50)
        ax4.set_title('Compression Ratio Distribution')
        ax4.set_xlabel('Compression Ratio')
        ax4.set_ylabel('Count')
        ax4.grid(True)
        
        # Save combined plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_stats.png'))
        plt.close()
        
        # Save individual plots
        self._save_individual_plots(
            iterations, vocab_sizes, compression_ratios, token_freqs
        )
    
    def _save_individual_plots(self, iterations, vocab_sizes, compression_ratios, token_freqs):
        """Save individual plots for each metric."""
        # Vocabulary Size
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, vocab_sizes)
        plt.title('Vocabulary Size Growth')
        plt.xlabel('Iteration')
        plt.ylabel('Vocabulary Size')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'vocab_size.png'))
        plt.close()
        
        # Compression Ratio
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, compression_ratios)
        plt.title('Compression Ratio Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Compression Ratio')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'compression_ratio.png'))
        plt.close()
        
        # Token Frequencies
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, token_freqs)
        plt.title('Token Frequencies')
        plt.xlabel('Iteration')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'token_frequencies.png'))
        plt.close() 