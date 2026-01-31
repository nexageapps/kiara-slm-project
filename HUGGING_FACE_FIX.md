# Fixing "ModuleNotFoundError: No module named 'kiara'" on Hugging Face Spaces

## Problem
Your Hugging Face Space is failing with:
```
ModuleNotFoundError: No module named 'kiara'
```

## Root Cause
The `src/kiara/` directory structure is missing from your Hugging Face Space. The app needs the entire source code to be uploaded.

## Solution

### Quick Fix (Manual Upload)

1. **Go to your Hugging Face Space**
   - Navigate to: https://huggingface.co/spaces/nexageapps/kiara-slm-project

2. **Upload the src directory**
   - Click "Files" tab
   - Click "Add file" → "Upload files"
   - Upload the entire `src/` directory from your local project
   - Make sure the structure looks like this:
     ```
     /
     ├── app.py
     ├── requirements.txt
     ├── README.md
     └── src/
         └── kiara/
             ├── __init__.py
             ├── model.py
             ├── training.py
             ├── attention.py
             ├── config.py
             ├── tokenizer.py
             └── utils/
                 ├── __init__.py
                 ├── checkpoint.py
                 ├── logging.py
                 └── metrics.py
     ```

3. **Verify the upload**
   - Check that all files are present in the Files tab
   - The Space should automatically rebuild

4. **Wait for rebuild**
   - The Space will rebuild automatically (2-5 minutes)
   - Check the build logs for any errors

### Automated Fix (Using Deploy Script)

Run the deployment script from your local project:

```bash
# Make sure you're logged into Hugging Face
huggingface-cli login

# Run the deploy script
./scripts/deploy_to_hf.sh
```

This script will:
- Copy all necessary files to a temporary directory
- Push them to your Hugging Face Space
- Maintain the correct directory structure

## What I Fixed

I've updated the following files to handle imports better:

1. **hf_spaces/app.py** - Added fallback import logic that:
   - First tries to import `kiara` directly
   - If that fails, looks for `src/` directory in the current or parent directory
   - Adds the appropriate path to `sys.path`

2. **hf_spaces/requirements.txt** - Kept it simple with just the dependencies (no `-e .`)

## Verification Checklist

After uploading, verify:

- [ ] `src/kiara/` directory exists in your Space
- [ ] All Python files are present in `src/kiara/`
- [ ] `app.py` is at the root level
- [ ] `requirements.txt` is at the root level
- [ ] Space builds without errors
- [ ] App loads and shows the interface
- [ ] (Optional) Model checkpoint is uploaded to `checkpoints/best_model.pt`

## Testing

Once deployed, test the Space:

1. Open your Space URL
2. Check if the interface loads
3. Try generating text with a simple prompt
4. Verify no import errors in the logs

## Alternative: Use Git Push

If you prefer using Git:

```bash
# Clone your Space
git clone https://huggingface.co/spaces/nexageapps/kiara-slm-project kiara-hf-space
cd kiara-hf-space

# Copy files from your project
cp /path/to/your/project/hf_spaces/app.py .
cp /path/to/your/project/hf_spaces/requirements.txt .
cp /path/to/your/project/hf_spaces/README.md .
cp -r /path/to/your/project/src .

# Commit and push
git add .
git commit -m "Add src directory to fix import error"
git push
```

## Need More Help?

If you're still having issues:

1. Check the Space logs (click "Logs" tab in your Space)
2. Verify the file structure matches the expected layout
3. Make sure all `__init__.py` files are present
4. Check that there are no syntax errors in the Python files

## Summary

The fix is simple: **upload the `src/` directory to your Hugging Face Space**. The app needs access to the source code to import the `kiara` module.
