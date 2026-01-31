# Fixing "No model loaded" Error on Hugging Face Spaces

## The Problem

Your Hugging Face Space shows:
```
⚠️ No model loaded
Please upload a checkpoint file or set CHECKPOINT_PATH environment variable.
```

This happens because the app can't find a trained model checkpoint file.

## Quick Fix Options

### Option 1: Upload Checkpoint via Web Interface (Easiest - 2 minutes)

1. **Train a model first** (if you haven't already):
   ```bash
   python scripts/train.py --config configs/small.yaml
   ```
   This creates `checkpoints/best_model.pt`

2. **Go to your Space**: https://huggingface.co/spaces/nexageapps/kiara-slm-project

3. **Click "Files" tab** at the top

4. **Create checkpoints folder**:
   - Click "Add file" → "Create a new file"
   - Name it: `checkpoints/.gitkeep`
   - Click "Commit new file"

5. **Upload your checkpoint**:
   - Click "Add file" → "Upload files"
   - Navigate to `checkpoints` folder
   - Drag and drop your `best_model.pt` file
   - Click "Commit changes"

6. **Wait for rebuild** (2-3 minutes)
   - The Space will automatically rebuild
   - Refresh the page once it's done

### Option 2: Deploy with Script (Automated)

If you have a checkpoint locally, use the deploy script:

```bash
# Make sure you have a checkpoint
ls checkpoints/best_model.pt

# Deploy (includes checkpoint automatically)
./scripts/deploy_to_hf.sh
```

The script will:
- Copy your app files
- Include the checkpoint if it exists
- Push everything to HF Spaces

### Option 3: Use Git LFS (For large checkpoints)

If your checkpoint is very large (>100MB):

1. **Install Git LFS**:
   ```bash
   # macOS
   brew install git-lfs
   
   # Linux
   apt-get install git-lfs
   
   # Initialize
   git lfs install
   ```

2. **Clone your Space**:
   ```bash
   git clone https://huggingface.co/spaces/nexageapps/kiara-slm-project
   cd kiara-slm-project
   ```

3. **Track checkpoint files with LFS**:
   ```bash
   git lfs track "*.pt"
   git add .gitattributes
   ```

4. **Add your checkpoint**:
   ```bash
   mkdir -p checkpoints
   cp /path/to/your/best_model.pt checkpoints/
   git add checkpoints/best_model.pt
   git commit -m "Add model checkpoint"
   git push
   ```

### Option 4: Use Environment Variable (Advanced)

If your checkpoint is hosted elsewhere:

1. **Go to Space Settings**: https://huggingface.co/spaces/nexageapps/kiara-slm-project/settings

2. **Scroll to "Repository secrets"**

3. **Add new variable**:
   - Name: `CHECKPOINT_PATH`
   - Value: `/path/to/your/checkpoint.pt`

4. **Restart the Space**

## Verifying the Fix

After uploading the checkpoint:

1. Wait for the Space to rebuild (check the "Building" indicator)
2. Refresh the page
3. You should see "✅ Loaded" in the Model Info section
4. Try generating text with a prompt

## Troubleshooting

### "Checkpoint file is corrupted"

Your checkpoint might be incomplete. Re-train or re-upload:
```bash
python scripts/train.py --config configs/small.yaml
```

### "File too large to upload"

Use Git LFS (Option 3 above) or reduce model size:
```bash
# Train a smaller model
python scripts/train.py --config configs/small.yaml
```

### "Still showing 'No model loaded'"

Check the Space logs:
1. Go to your Space
2. Click "Logs" tab
3. Look for error messages about checkpoint loading

Common issues:
- Wrong file path (should be `checkpoints/best_model.pt`)
- File not committed properly
- Checkpoint format mismatch

### "Build failed after uploading"

The checkpoint might be too large for the free tier:
- Free tier: 50GB storage limit
- Consider using a smaller model
- Or upgrade to a paid tier

## Creating a Checkpoint (If You Don't Have One)

If you don't have a trained model yet:

```bash
# Quick training (5-10 minutes on CPU)
python scripts/train.py --config configs/small.yaml --epochs 5

# This creates: checkpoints/best_model.pt
```

Then use Option 1 or 2 above to upload it.

## Need Help?

- Check the full deployment guide: `documentation/HUGGING_FACE_SPACES.md`
- Open an issue: https://github.com/nexageapps/kiara-slm-project/issues
- HF Spaces docs: https://huggingface.co/docs/hub/spaces

## Summary

The error happens because there's no model checkpoint in your Space. Fix it by:
1. Training a model locally (if needed)
2. Uploading `best_model.pt` to `checkpoints/` folder in your Space
3. Waiting for the Space to rebuild

That's it! Your Space should work after the checkpoint is uploaded.
