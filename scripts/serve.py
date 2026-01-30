"""Simple API server for Kiara SLM."""

import argparse
import torch
import tiktoken
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from kiara.model import GPTModel
from kiara.training import generate_text_sampling
from kiara.utils import setup_logger


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = None


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str


# Global variables
model = None
tokenizer = None
device = None
model_config = None


def load_model(checkpoint_path: str, device_name: str = "cuda"):
    """Load model from checkpoint."""
    global model, tokenizer, device, model_config
    
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model_config = checkpoint.get('config', {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
    })
    
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")


# Create FastAPI app
app = FastAPI(title="Kiara SLM API", version="0.1.0")


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Kiara SLM API", "version": "0.1.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate text from prompt."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode prompt
        encoded = tokenizer.encode(request.prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            token_ids = generate_text_sampling(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=request.max_tokens,
                context_size=model_config["context_length"],
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # Decode
        generated_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Serve Kiara SLM API")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    """Main server function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name="kiara.serve", log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("Kiara SLM API Server")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    load_model(args.checkpoint, args.device)
    logger.info("Model loaded successfully")
    
    # Start server
    logger.info(f"\nStarting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
