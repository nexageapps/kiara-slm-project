#!/bin/bash
# Deploy Kiara SLM to Hugging Face Spaces
# Usage: ./scripts/deploy_to_hf.sh
# Prerequisites: git, Hugging Face CLI (huggingface_hub) with login

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_REPO="nexageapps/kiara-slm-project"
HF_URL="https://huggingface.co/spaces/${HF_REPO}"

echo "🚀 Deploying Kiara SLM to Hugging Face Spaces"
echo "   Target: $HF_URL"
echo ""

# Use temp dir to avoid creating kiara-slm-project/kiara-slm-project
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "📥 Cloning Hugging Face Space..."
cd "$TEMP_DIR"
git clone "$HF_URL" space
cd space

echo "📋 Copying deployment files..."
cp "$PROJECT_ROOT/hf_spaces/app.py" .
cp "$PROJECT_ROOT/hf_spaces/requirements.txt" .
cp "$PROJECT_ROOT/hf_spaces/README.md" .
rm -rf src
cp -r "$PROJECT_ROOT/src" .

# Copy checkpoint if it exists
if [ -d "$PROJECT_ROOT/checkpoints" ] && [ -f "$PROJECT_ROOT/checkpoints/best_model.pt" ]; then
  mkdir -p checkpoints
  cp "$PROJECT_ROOT/checkpoints/best_model.pt" checkpoints/
  echo "   ✅ Checkpoint included"
else
  echo "   ⚠️  No checkpoint found (optional - Space will work without it)"
fi

echo "📤 Pushing to Hugging Face..."
# Configure git identity for commit (required if not set globally)
git config user.email "deploy@kiara-slm.local"
git config user.name "Kiara SLM Deploy"

git add .
if git diff --staged --quiet; then
  echo "   No changes to push"
else
  git commit -m "Deploy: Sync from local ($(date +%Y-%m-%d))"
  git push
  echo ""
  echo "✅ Successfully deployed to Hugging Face Spaces!"
fi

echo ""
echo "🌐 View your Space: https://huggingface.co/spaces/$HF_REPO"
