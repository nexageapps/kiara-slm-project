# API Documentation

## REST API Server

Kiara SLM provides a FastAPI-based REST API for text generation.

## Starting the Server

```bash
python scripts/serve.py \
    --checkpoint checkpoints/best_model.pt \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda
```

## Endpoints

### GET /

Root endpoint with API information.

**Response:**
```json
{
  "message": "Kiara SLM API",
  "version": "0.1.0"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /generate

Generate text from a prompt.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 50
}
```

**Parameters:**
- `prompt` (string, required): Input text prompt
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 100)
- `temperature` (float, optional): Sampling temperature (default: 0.8)
- `top_k` (integer, optional): Top-k sampling parameter (default: null)

**Response:**
```json
{
  "generated_text": "Once upon a time, in a land far away...",
  "prompt": "Once upon a time"
}
```

## Usage Examples

### cURL

```bash
# Generate text
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The future of AI",
        "max_tokens": 50,
        "temperature": 0.8
    }'

# Health check
curl http://localhost:8000/health
```

### Python

```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "The future of AI",
        "max_tokens": 50,
        "temperature": 0.8,
        "top_k": 50
    }
)

result = response.json()
print(result["generated_text"])
```

### JavaScript

```javascript
// Generate text
fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        prompt: 'The future of AI',
        max_tokens: 50,
        temperature: 0.8
    })
})
.then(response => response.json())
.then(data => console.log(data.generated_text));
```

## Docker Deployment

```bash
# Using docker-compose
docker-compose up api

# Or directly
docker run -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kiara-slm:latest \
    python scripts/serve.py --checkpoint /app/checkpoints/best_model.pt
```

## Error Handling

### 503 Service Unavailable

Model not loaded. Check server logs.

### 500 Internal Server Error

Generation failed. Check request parameters and server logs.

## Performance Tips

1. Use GPU for faster inference
2. Batch requests when possible
3. Adjust `max_tokens` based on needs
4. Use lower `temperature` for more deterministic output
5. Enable model caching for repeated requests
