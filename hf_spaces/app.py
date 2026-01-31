"""
Gradio interface for Kiara SLM on Hugging Face Spaces.
This app allows users to test the Kiara Small Language Model without local setup.
"""

import gradio as gr
import torch
import tiktoken
import os
import sys
from pathlib import Path

# Handle imports - try direct import first, then add src to path if needed
try:
    from kiara.model import GPTModel
    from kiara.training import generate_text_simple, generate_text_sampling
except ModuleNotFoundError:
    # If kiara module not found, add src directory to path
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    else:
        # Try parent directory's src
        src_path = Path(__file__).parent.parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
    
    from kiara.model import GPTModel
    from kiara.training import generate_text_simple, generate_text_sampling


class KiaraSpacesApp:
    """Gradio application for Kiara SLM."""
    
    def __init__(self):
        """Initialize the app with model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer (may fail in offline environments)
        try:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize tokenizer: {e}")
            print("   This is expected in offline environments.")
            self.tokenizer = None
        
        self.model = None
        self.model_config = None
        self.checkpoint_path = None
        
        # Try to load a checkpoint if available
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load model checkpoint if available."""
        # Look for checkpoint in several possible locations
        possible_paths = [
            "checkpoints/best_model.pt",
            "../checkpoints/best_model.pt",
            "checkpoint.pt",
            os.environ.get("CHECKPOINT_PATH", ""),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                self.checkpoint_path = path
                break
        
        if self.checkpoint_path:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                # Extract model config
                self.model_config = checkpoint.get('config', {
                    "vocab_size": 50257,
                    "context_length": 256,
                    "emb_dim": 768,
                    "n_heads": 12,
                    "n_layers": 12,
                    "drop_rate": 0.1,
                })
                
                # Create and load model
                self.model = GPTModel(self.model_config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Model loaded successfully from {self.checkpoint_path}")
                print(f"📊 Device: {self.device}")
                print(f"🔧 Config: {self.model_config}")
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                self.model = None
        else:
            print("⚠️ No checkpoint found. Please upload a checkpoint to use the model.")
    
    def get_model_info(self):
        """Get model information as formatted string."""
        if self.model is None:
            return "⚠️ **No model loaded**\n\nPlease upload a checkpoint file or set CHECKPOINT_PATH environment variable."
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = f"""
### 🤖 Model Information

**Status:** ✅ Loaded  
**Checkpoint:** `{self.checkpoint_path}`  
**Device:** `{self.device}`

**Architecture:**
- Vocabulary Size: {self.model_config['vocab_size']:,}
- Context Length: {self.model_config['context_length']:,}
- Embedding Dimension: {self.model_config['emb_dim']:,}
- Number of Heads: {self.model_config['n_heads']}
- Number of Layers: {self.model_config['n_layers']}
- Dropout Rate: {self.model_config['drop_rate']}

**Parameters:**
- Total: {total_params:,}
- Trainable: {trainable_params:,}
        """
        return info
    
    def generate_text(self, prompt, max_tokens, temperature, top_k, use_sampling):
        """
        Generate text based on the prompt and parameters.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            use_sampling: Whether to use sampling (True) or greedy decoding (False)
        
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            return "❌ Error: Tokenizer not initialized. This may be due to network issues."
        
        if self.model is None:
            return "❌ Error: No model loaded. Please upload a checkpoint first."
        
        if not prompt or not prompt.strip():
            return "❌ Error: Please provide a text prompt."
        
        try:
            # Encode the prompt
            encoded = self.tokenizer.encode(prompt)
            encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(self.device)
            
            # Generate text
            with torch.no_grad():
                if use_sampling and temperature > 0:
                    # Sampling with temperature and optional top-k
                    token_ids = generate_text_sampling(
                        model=self.model,
                        idx=encoded_tensor,
                        max_new_tokens=max_tokens,
                        context_size=self.model_config["context_length"],
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None
                    )
                else:
                    # Greedy decoding
                    token_ids = generate_text_simple(
                        model=self.model,
                        idx=encoded_tensor,
                        max_new_tokens=max_tokens,
                        context_size=self.model_config["context_length"]
                    )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(token_ids.squeeze(0).tolist())
            return generated_text
            
        except Exception as e:
            return f"❌ Error during generation: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface."""
    app = KiaraSpacesApp()
    
    with gr.Blocks(title="Kiara SLM - Small Language Model") as demo:
        gr.Markdown("""
        # 🚀 Kiara - Small Language Model
        
        An open-source, production-ready Small Language Model built from scratch.
        This demo allows you to interact with Kiara and generate text using various sampling strategies.
        
        **Features:**
        - 🎯 Adjustable temperature and top-k sampling
        - 🔄 Greedy and sampling-based generation
        - 📊 Real-time model information
        - 🎨 Interactive parameter tuning
        
        **GitHub:** [nexageapps/kiara-slm-project](https://github.com/nexageapps/kiara-slm-project)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Text Generation")
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=3,
                    value="Once upon a time"
                )
                
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Tokens",
                        info="Maximum number of tokens to generate"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more random, 0 = greedy"
                    )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-K",
                        info="0 = disabled, higher = more diverse"
                    )
                    
                    use_sampling = gr.Checkbox(
                        label="Use Sampling",
                        value=True,
                        info="Enable temperature-based sampling"
                    )
                
                generate_btn = gr.Button("🎨 Generate Text", variant="primary", size="lg")
                
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    interactive=False
                )
                
                gr.Markdown("""
                ### 💡 Tips
                - **Temperature 0**: Greedy decoding (deterministic, always picks most likely token)
                - **Temperature 0.7-0.9**: Balanced creativity and coherence
                - **Temperature > 1.0**: More random and creative (but less coherent)
                - **Top-K**: Limits sampling to K most likely tokens (improves quality)
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Model Info")
                model_info = gr.Markdown(app.get_model_info())
                
                refresh_btn = gr.Button("🔄 Refresh Info", size="sm")
                
                gr.Markdown("""
                ### 📚 About Kiara
                
                Kiara is an educational implementation of a Small Language Model (SLM) 
                built from scratch to demonstrate how transformer-based language models work.
                
                **Key Features:**
                - GPT-style decoder-only architecture
                - Multi-head self-attention
                - Positional embeddings
                - Layer normalization
                - Residual connections
                
                **Use Cases:**
                - Learning how LLMs work
                - Prototyping language model applications
                - Educational demonstrations
                - Research experiments
                
                ### 🔗 Resources
                - [GitHub Repository](https://github.com/nexageapps/kiara-slm-project)
                - [Documentation](https://github.com/nexageapps/kiara-slm-project/tree/main/documentation)
                - [Quick Start Guide](https://github.com/nexageapps/kiara-slm-project/blob/main/documentation/QUICKSTART.md)
                """)
        
        # Event handlers
        generate_btn.click(
            fn=app.generate_text,
            inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, use_sampling],
            outputs=output_text
        )
        
        refresh_btn.click(
            fn=app.get_model_info,
            outputs=model_info
        )
        
        gr.Markdown("""
        ---
        ### 📄 License
        
        MIT License - Free to use, modify, and distribute.
        
        **Built by the open-source community** | [Sponsor on GitHub](https://github.com/sponsors/nexageapps)
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
