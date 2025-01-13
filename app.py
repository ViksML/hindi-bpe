import gradio as gr
import os
from bpe import HindiBPE
import json

# Load the trained model
MODEL_DIR = os.path.join('models', 'hindi_bpe')
STATS_DIR = os.path.join('stats', 'hindi_bpe')

def load_model():
    """Load the trained BPE model."""
    model = HindiBPE(vocab_size=5000)
    model_path = os.path.join(MODEL_DIR, 'model.json')
    stats_path = os.path.join(STATS_DIR, 'metrics.json')
    
    try:
        model.load(model_path, stats_path)
        return model
    except FileNotFoundError:
        raise Exception("Model not found. Please train the model first using train_hindi_bpe.py")

def tokenize_text(text, model):
    """Tokenize input text and return tokens with statistics."""
    if not text.strip():
        return "Please enter some text.", "", ""
    
    # Encode text
    tokens = model.encode(text)
    
    # Get statistics
    original_chars = len(text)
    encoded_tokens = len(tokens)
    compression_ratio = original_chars / encoded_tokens if encoded_tokens > 0 else 0
    unique_tokens = len(set(tokens))
    
    # Format statistics
    stats = f"""
    Statistics:
    - Original characters: {original_chars}
    - Encoded tokens: {encoded_tokens}
    - Unique tokens used: {unique_tokens}
    - Compression ratio: {compression_ratio:.2f}
    - Average token length: {sum(len(t) for t in tokens) / len(tokens):.2f}
    """
    
    # Format tokens with details
    tokens_display = "\n".join([
        f"{i+1}. '{token}' (length: {len(token)})"
        for i, token in enumerate(tokens)
    ])
    
    # Decode back to text
    decoded_text = model.decode(tokens)
    
    return tokens_display, decoded_text, stats

def create_interface():
    """Create the Gradio interface."""
    # Load model
    model = load_model()
    
    # Define interface
    iface = gr.Interface(
        fn=lambda text: tokenize_text(text, model),
        inputs=[
            gr.Textbox(
                lines=5,
                placeholder="Enter Hindi text here...",
                label="Input Text"
            )
        ],
        outputs=[
            gr.Textbox(
                label="Tokens (with details)",
                lines=10,
                placeholder="Tokenization details will appear here..."
            ),
            gr.Textbox(label="Decoded Text"),
            gr.Textbox(label="Statistics")
        ],
        title="Hindi BPE Tokenizer",
        description="""
        This application demonstrates Byte-Pair Encoding (BPE) tokenization for Hindi text.
        Enter Hindi text in the input box to see its tokenized form and statistics.
        
        Model Details:
        - Vocabulary Size: 5000 tokens
        - Target Compression Ratio: > 3.5
        
        Token Display Format:
        - Each token is shown with its position and length
        - Statistics show total, unique tokens and compression metrics
        """,
        examples=[
            ["नमस्ते भारत। यह एक परीक्षण वाक्य है।"],
            ["हिंदी भाषा बहुत सुंदर है।"],
            ["मैं भारत से प्यार करता हूं।"]
        ],
        theme=gr.themes.Soft()
    )
    return iface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    ) 