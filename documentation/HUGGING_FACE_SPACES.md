# Hugging Face Spaces Deployment Guide

Complete guide for deploying Kiara SLM to Hugging Face Spaces.

## 📖 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Methods](#deployment-methods)
- [Configuration](#configuration)
- [Testing](#testing)
- [Sharing & Embedding](#sharing--embedding)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

Hugging Face Spaces provides a free platform to deploy and share machine learning applications with a simple Gradio interface. This guide walks you through deploying Kiara SLM to Spaces.

### Benefits

✅ **Free Hosting** - Free tier with CPU, upgraded tier with GPU  
✅ **Zero Setup** - No server configuration needed  
✅ **Easy Sharing** - Public URL instantly available  
✅ **Embedded Player** - Embed in websites and blogs  
✅ **API Access** - Automatic API generation  
✅ **GitHub Sync** - Auto-deploy from GitHub

## Prerequisites

### Required

1. **Hugging Face Account**
   - Sign up at [huggingface.co/join](https://huggingface.co/join)
   - Free account is sufficient

2. **Trained Model Checkpoint**
   - Train a model using: `python scripts/train.py --config configs/small.yaml`
   - Or use an existing checkpoint from `checkpoints/best_model.pt`

### Optional

- **Git** (for GitHub sync method)
- **Hugging Face Token** (for automated deployments)
- **Python 3.8+** (for local testing)

## Quick Start

### 5-Minute Deploy (Manual Upload)

1. **Create a new Space**
   ```
   Navigate to: https://huggingface.co/spaces
   Click: "Create new Space"
   ```

2. **Configure Space**
   - **Owner:** Your username
   - **Space name:** `kiara-slm` (or your choice)
   - **License:** MIT
   - **Space SDK:** Select **Gradio**
   - **Visibility:** Public (or Private)
   - **Space hardware:** CPU basic (Free) or GPU (Upgraded)

3. **Clone and prepare files**
   ```bash
   # Clone the repo
   git clone https://github.com/nexageapps/kiara-slm-project.git
   cd kiara-slm-project
   ```

4. **Upload to your Space**
   - Go to your newly created Space on Hugging Face
   - Click "Files" tab
   - Upload the following files:
     - `hf_spaces/app.py` → Upload as `app.py` (root level)
     - `hf_spaces/requirements.txt` → Upload as `requirements.txt` (root level)
   - Create `src/` folder and upload entire `src/kiara/` directory
   - Create `checkpoints/` folder and upload your `best_model.pt`

5. **Wait for build**
   - Space will automatically build (takes 2-5 minutes)
   - Check the build logs for any errors

6. **Test your Space**
   - Once built, the app should be live
   - Enter a prompt and click "Generate Text"

## Deployment Methods

### Method 1: Manual Upload (Recommended for Beginners)

**Pros:** Simple, visual, no CLI needed  
**Cons:** Manual updates required

**Steps:**

1. Create Space as shown in Quick Start
2. Use Hugging Face web interface to upload files
3. Manually update files when needed

### Method 2: Git Push (Recommended for Development)

**Pros:** Version control, easy updates  
**Cons:** Requires Git knowledge

**Steps:**

1. **Create Space with Git**
   ```bash
   # Create space on HF website first, then clone
   git clone https://huggingface.co/spaces/YOUR_USERNAME/kiara-slm
   cd kiara-slm
   ```

2. **Copy files**
   ```bash
   # Copy from your kiara-slm-project
   cp path/to/kiara-slm-project/hf_spaces/app.py .
   cp path/to/kiara-slm-project/hf_spaces/requirements.txt .
   cp -r path/to/kiara-slm-project/src .
   mkdir checkpoints
   cp path/to/kiara-slm-project/checkpoints/best_model.pt checkpoints/
   ```

3. **Commit and push**
   ```bash
   git add .
   git commit -m "Initial deployment of Kiara SLM"
   git push
   ```

4. **Monitor build**
   - Watch the Space page for build status

### Method 3: GitHub Sync (Automated, Recommended for CI/CD)

**Pros:** Automatic deployments, CI/CD integration  
**Cons:** More setup required

**Steps:**

1. **Create Hugging Face Token**
   ```
   Go to: https://huggingface.co/settings/tokens
   Click: "New token"
   Name: "GitHub Actions"
   Role: "Write"
   Copy the token
   ```

2. **Add token to GitHub**
   ```
   Go to: Your repo → Settings → Secrets and variables → Actions
   Click: "New repository secret"
   Name: HF_TOKEN
   Value: Your HF token
   ```

3. **Configure workflow**
   - The `.github/workflows/hf-spaces-sync.yml` is already set up
   - Edit to add your Space username/name:
   ```yaml
   env:
     HF_USERNAME: your-hf-username
     HF_SPACE_NAME: kiara-slm
   ```

4. **Enable workflow**
   ```bash
   git add .github/workflows/hf-spaces-sync.yml
   git commit -m "Enable HF Spaces sync"
   git push
   ```

5. **Trigger deployment**
   - Push changes to `main` branch
   - Or manually trigger via GitHub Actions tab

## Configuration

### Model Checkpoint

The app searches for checkpoints in this order:
1. `checkpoints/best_model.pt` (relative to app.py)
2. `../checkpoints/best_model.pt`
3. `checkpoint.pt`
4. Path from `CHECKPOINT_PATH` environment variable

### Environment Variables

Set in Space Settings → Variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CHECKPOINT_PATH` | Path to checkpoint | `checkpoints/best_model.pt` | `/app/models/model.pt` |
| `DEVICE` | Force device | Auto-detect | `cuda` or `cpu` |

**To set environment variables:**
1. Go to your Space page
2. Click Settings
3. Scroll to "Repository secrets"
4. Add variable name and value

### Hardware Selection

#### CPU (Free Tier)
- **Cost:** Free
- **Memory:** 16GB RAM
- **Speed:** Slower (5-10s per generation)
- **Best for:** Demos, testing, low traffic

#### GPU (Upgraded Tier)
- **Cost:** Paid (varies by GPU type)
- **Options:** T4, A10G, A100
- **Speed:** Fast (0.5-2s per generation)
- **Best for:** Production, high traffic

**To upgrade:**
1. Go to Space Settings
2. Click "Change hardware"
3. Select desired GPU
4. Confirm billing

## Testing

### Local Testing (Before Deploy)

Test the Gradio app locally before deploying:

```bash
# Method 1: Direct run
cd hf_spaces
python app.py
# Visit http://localhost:7860

# Method 2: Testing script
python scripts/test_spaces.py
```

### Testing Checklist

Before deploying, verify:

- [ ] Model checkpoint loads successfully
- [ ] All imports work correctly
- [ ] Gradio interface displays properly
- [ ] Text generation works with sample prompts
- [ ] Parameter sliders function correctly
- [ ] Model info displays accurate information
- [ ] No error messages in console

## Sharing & Embedding

### Direct Link

Share your Space URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/kiara-slm
```

### Embed in Website

Use Gradio's embed code:

```html
<gradio-app src="https://YOUR_USERNAME-kiara-slm.hf.space"></gradio-app>
<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/[VERSION]/gradio.js"></script>
```

**Note:** Replace `[VERSION]` with the current Gradio version (check your Space's settings or use the embed code provided by Hugging Face).

### API Access

Hugging Face provides automatic API access:

**Python:**
```python
from gradio_client import Client

client = Client("YOUR_USERNAME/kiara-slm")
result = client.predict(
    "Once upon a time",  # prompt
    100,                  # max_tokens
    0.8,                  # temperature
    50,                   # top_k
    True,                 # use_sampling
    api_name="/predict"
)
print(result)
```

**cURL:**
```bash
curl -X POST "https://YOUR_USERNAME-kiara-slm.hf.space/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": ["Once upon a time", 100, 0.8, 50, true]
  }'
```

**JavaScript:**
```javascript
const response = await fetch(
  "https://YOUR_USERNAME-kiara-slm.hf.space/api/predict",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: ["Once upon a time", 100, 0.8, 50, true]
    })
  }
);
const result = await response.json();
console.log(result.data);
```

### Social Sharing

Share on social media using these badges:

**Markdown:**
```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/YOUR_USERNAME/kiara-slm)
```

**HTML:**
```html
<a href="https://huggingface.co/spaces/YOUR_USERNAME/kiara-slm">
  <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg" alt="Open in Spaces">
</a>
```

## Troubleshooting

### Common Issues

#### 1. "No model loaded" Error

**Symptoms:** Error message saying no model is available

**Solutions:**
- Verify checkpoint file is uploaded to `checkpoints/best_model.pt`
- Check checkpoint file size (should be > 1MB)
- Try setting `CHECKPOINT_PATH` environment variable
- Ensure checkpoint was saved correctly during training

#### 2. Build Failures

**Symptoms:** Space stuck in "Building" or shows build errors

**Solutions:**
- Check requirements.txt has all dependencies
- View build logs for specific errors
- Ensure app.py has no syntax errors
- Verify Python version compatibility (3.8+)

**Common build errors:**
```
ModuleNotFoundError: No module named 'kiara'
→ Ensure src/ directory is uploaded

ModuleNotFoundError: No module named 'gradio'
→ Check requirements.txt includes gradio>=4.0.0

ImportError: cannot import name 'GPTModel'
→ Verify src/kiara/model.py is present
```

#### 3. CUDA Out of Memory

**Symptoms:** Generation fails with CUDA memory error

**Solutions:**
- Reduce max_tokens slider value
- Switch to CPU in Settings → Hardware
- Use a smaller model checkpoint
- Restart the Space

#### 4. Slow Generation

**Symptoms:** Text generation takes very long

**Solutions:**
- Upgrade to GPU hardware (Settings → Hardware)
- Reduce max_tokens parameter
- Use greedy decoding (temperature=0) instead of sampling
- Check if using CPU instead of GPU

#### 5. Model Generation Quality Issues

**Symptoms:** Generated text is incoherent or repetitive

**Solutions:**
- Verify model was trained properly
- Try adjusting temperature (0.7-0.9 is usually good)
- Enable top-k sampling (try k=50)
- Check if model checkpoint is corrupted
- Train for more epochs if needed

### Debug Mode

Enable debug logging:

```python
# In app.py, add at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then check Space logs for detailed information.

### Getting Help

If you're still stuck:

1. **Check Space logs**
   - Click "Logs" tab in your Space
   - Look for error messages

2. **Review documentation**
   - [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
   - [Gradio Documentation](https://gradio.app/docs/)

3. **Community support**
   - [Hugging Face Forums](https://discuss.huggingface.co/)
   - [GitHub Issues](https://github.com/nexageapps/kiara-slm-project/issues)

## Advanced Topics

### Custom Domain

Set up a custom domain for your Space:

1. Go to Space Settings
2. Click "Custom domain"
3. Follow instructions to add DNS records
4. Verify domain ownership

### Authentication

Add password protection:

```python
# In app.py, modify launch():
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    auth=("username", "password")
)
```

### Analytics

Track usage with Gradio analytics:

```python
# In app.py:
demo.launch(
    analytics_enabled=True
)
```

View analytics in Space Settings → Analytics.

### Model Caching

For faster loading, cache the model:

```python
import functools

@functools.lru_cache(maxsize=1)
def load_model():
    # Model loading code
    return model
```

### Multiple Checkpoints

Support multiple model sizes:

```python
checkpoint_selector = gr.Dropdown(
    choices=["small", "medium", "large"],
    label="Model Size"
)

def load_checkpoint(size):
    path = f"checkpoints/{size}_model.pt"
    # Load and return model
```

### Rate Limiting

Prevent abuse with rate limiting:

```python
# In app.py:
demo.launch(
    max_threads=10,  # Limit concurrent requests
)
```

Or use Hugging Face's built-in rate limiting in Settings.

## Best Practices

### Performance

1. **Use GPU** for production deployments
2. **Cache models** to reduce load time
3. **Limit max_tokens** to reasonable values (50-200)
4. **Implement timeout** for long generations
5. **Monitor resource usage** in Space settings

### Security

1. **Don't commit secrets** in code
2. **Use environment variables** for sensitive data
3. **Enable authentication** for private demos
4. **Validate user inputs** to prevent injection
5. **Rate limit** to prevent abuse

### User Experience

1. **Provide clear instructions** in the interface
2. **Set reasonable defaults** for parameters
3. **Show loading indicators** during generation
4. **Handle errors gracefully** with user-friendly messages
5. **Include examples** and tips

### Maintenance

1. **Monitor Space logs** regularly
2. **Update dependencies** periodically
3. **Test changes locally** before deploying
4. **Keep checkpoint backups**
5. **Document changes** in README

## Resources

### Documentation
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Gradio Guide](https://gradio.app/guides/)
- [Kiara SLM Docs](https://github.com/nexageapps/kiara-slm-project/tree/main/documentation)

### Examples
- [Hugging Face Spaces Gallery](https://huggingface.co/spaces)
- [Gradio Examples](https://gradio.app/demos/)

### Community
- [HF Discord](https://discord.gg/hugging-face)
- [HF Forums](https://discuss.huggingface.co/)
- [GitHub Issues](https://github.com/nexageapps/kiara-slm-project/issues)

## Next Steps

After deploying your Space:

1. ✅ Test all functionality
2. ✅ Share with community
3. ✅ Monitor usage and performance
4. ✅ Gather user feedback
5. ✅ Iterate and improve

**Happy deploying! 🚀**

---

**Need help?** Open an issue on [GitHub](https://github.com/nexageapps/kiara-slm-project/issues)
