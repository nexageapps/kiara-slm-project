---
title: Kiara SLM
emoji: 🚀
sdk: gradio
sdk_version: "4.0.0"
---

# Kiara SLM - Hugging Face Spaces

This directory contains the Gradio application for deploying Kiara SLM on Hugging Face Spaces.

## 🚀 Quick Deploy to Hugging Face Spaces

### Prerequisites
- A Hugging Face account ([Sign up here](https://huggingface.co/join))
- A trained Kiara model checkpoint

### Deployment Steps

#### Option 1: Direct Upload (Easiest)

1. **Create a new Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose a name (e.g., `my-kiara-slm`)
   - Select **Gradio** as the SDK
   - Choose **Public** or **Private**
   - Click "Create Space"

2. **Upload files**
   - Upload `app.py` to the root of your Space
   - Upload `requirements.txt` to the root
   - Create a `src/` directory and upload the `kiara/` package from this repo

3. **Upload your model checkpoint**
   - Upload your trained `best_model.pt` to a `checkpoints/` directory in your Space
   - Or set the `CHECKPOINT_PATH` environment variable to point to your checkpoint

4. **Wait for build**
   - Hugging Face will automatically build and deploy your Space
   - This takes 2-5 minutes

5. **Done!** 🎉
   - Your Space will be live at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

#### Option 2: GitHub Sync (Automated)

1. **Create a new Space with GitHub sync**
   - Create a Space as in Option 1
   - Enable "Sync with GitHub" in Space settings
   - Connect to this repository

2. **Use GitHub Actions workflow**
   - The `.github/workflows/hf-spaces-sync.yml` workflow will automatically sync changes
   - Set up the `HF_TOKEN` secret in your GitHub repository settings

3. **Configure environment**
   - In your Space settings, add any required secrets/environment variables
   - Set `CHECKPOINT_PATH` if needed

## 📁 Files

- **`app.py`** - Main Gradio application with interactive interface
- **`requirements.txt`** - Python dependencies for Hugging Face Spaces
- **`README.md`** - This file

## 🎨 Features

The Gradio interface includes:

- **Interactive Text Generation** - Generate text from custom prompts
- **Parameter Controls**
  - Temperature slider (0.0 - 2.0)
  - Top-K sampling (0 - 100)
  - Max tokens (10 - 500)
  - Sampling mode toggle
- **Model Information Display** - View architecture and parameter counts
- **Real-time Generation** - See results immediately

## 🔧 Configuration

### Environment Variables

You can configure the app using environment variables in your Space settings:

- `CHECKPOINT_PATH` - Path to model checkpoint (default: `checkpoints/best_model.pt`)
- `DEVICE` - Force device selection (`cuda` or `cpu`, auto-detected by default)

### Model Checkpoint

The app looks for checkpoints in the following order:
1. `checkpoints/best_model.pt`
2. `../checkpoints/best_model.pt`
3. `checkpoint.pt`
4. Path specified in `CHECKPOINT_PATH` environment variable

## 🧪 Local Testing

Before deploying to Hugging Face Spaces, test the app locally:

```bash
# From the repository root
cd hf_spaces
python app.py
```

The app will start at `http://localhost:7860`

Or use the testing script:

```bash
python scripts/test_spaces.py
```

## 📊 Hardware Requirements

### CPU (Free Tier)
- Works but slower
- Suitable for demo purposes
- Text generation takes several seconds

### GPU (Upgraded Tier)
- Much faster inference
- Recommended for production use
- Near real-time text generation

## 🔗 API Usage

Gradio automatically provides an API for your Space. You can interact with it programmatically:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/SPACE_NAME")
result = client.predict(
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    use_sampling=True,
    api_name="/predict"
)
print(result)
```

## 🚀 Sharing Your Space

### Embed in Websites

Hugging Face provides an embed code:

```html
<gradio-app src="https://YOUR_USERNAME-SPACE_NAME.hf.space"></gradio-app>
<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.40.0/gradio.js"></script>
```

### Share Direct Link

Share your Space URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
```

## 📚 Documentation

For complete deployment instructions, see:
- [HUGGING_FACE_SPACES.md](../documentation/HUGGING_FACE_SPACES.md) - Detailed deployment guide
- [Main README](../README.md) - Project documentation

## 🆘 Troubleshooting

### "No model loaded" error
- Ensure you've uploaded a checkpoint to `checkpoints/best_model.pt`
- Or set the `CHECKPOINT_PATH` environment variable
- Check the checkpoint file is valid (not corrupted)

### "CUDA out of memory" error
- Reduce max_tokens slider
- Use CPU instead of GPU
- Consider using a smaller model

### Slow generation
- Upgrade to GPU hardware in Space settings
- Reduce max_tokens
- Use greedy decoding (temperature=0) instead of sampling

### Build failures
- Check `requirements.txt` is complete
- Verify all imports in `app.py` are available
- Check Space logs for detailed error messages

## 🤝 Contributing

Found an issue or have a suggestion? Please open an issue in the main repository:
- [GitHub Issues](https://github.com/nexageapps/kiara-slm-project/issues)

## 📄 License

MIT License - Free to use, modify, and distribute.
