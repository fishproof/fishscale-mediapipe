# FishScale.AI — MediaPipe Face Landmark Service

A lightweight Python microservice that detects 478 facial landmarks using Google's MediaPipe Face Landmarker. Used by FishScale.AI to measure the angler's face in fishing photos for accurate fish weight estimation.

## What It Does

Accepts an image (URL or base64), detects the angler's face, and returns precise pixel measurements of facial features (temple-to-temple width, eye distance, face height). These measurements are used as a known-size reference to calculate the fish's length in inches.

## Endpoints

- `GET /health` — Health check (public)
- `POST /detect` — Detect face landmarks (requires API key)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | No | Server port (default: 8080, Railway sets this automatically) |
| `MEDIAPIPE_API_KEY` | Yes | API key for authenticating requests |

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/fishscale-mediapipe)

Or manually:
1. Fork this repo
2. Create a new Railway project → Deploy from GitHub
3. Set `MEDIAPIPE_API_KEY` environment variable
4. Railway auto-detects the Dockerfile and deploys

## Local Development

```bash
pip install -r requirements.txt
MEDIAPIPE_API_KEY=test python app.py
```

## API Usage

```bash
curl -X POST https://your-service.railway.app/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"image_url": "https://example.com/fishing-photo.jpg"}'
```
