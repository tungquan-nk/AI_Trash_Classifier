# AI Trash Classifier Backend

Standalone FastAPI service for image uploads. The API validates uploaded image
files, but classification is intentionally unavailable until the project team
confirms the production checkpoint and AI inference contract. No checkpoint is
loaded and no placeholder predictions are returned.

## Setup

From the repository root, create and activate a virtual environment, then run:

```bash
python -m venv .venv-backend
```

Windows PowerShell:

```powershell
.venv-backend\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

macOS/Linux:

```bash
source .venv-backend/bin/activate
python -m pip install -r backend/requirements.txt
```

## Run

From the repository root:

```bash
python -m uvicorn app.main:app --app-dir backend --reload
```

The API listens on `http://127.0.0.1:8000`. Interactive OpenAPI docs are at
`http://127.0.0.1:8000/docs`.

Set `MAX_UPLOAD_BYTES` to change the maximum accepted image size. The default
is 10 MiB. The API also limits the complete `/api/predict` request body to
`MAX_UPLOAD_BYTES + 1 MiB` for multipart framing and headers. This limit is
counted from actual incoming body chunks; `Content-Length` is only an early
rejection optimization. The body is bounded before FastAPI parses multipart
data and is spooled to a temporary file after 1 MiB in memory.

Decoded images are limited to 10,000 pixels per side and 25,000,000 pixels
total. Pillow fully decodes an image before the classifier service is called.
Empty, malformed, truncated, oversized, and over-dimension images are rejected.

### Reverse proxy example (Nginx)

The application enforces its own streaming request limit, including when
`Content-Length` is missing. Configure a matching or slightly larger limit at
the reverse proxy so oversized bodies are rejected before reaching the app.
With the defaults, Nginx's `11m` allows the 10 MiB file plus multipart overhead;
the application still enforces the stricter exact limit.

```nginx
location / {
    client_max_body_size 11m;
    client_body_timeout 15s;
    proxy_read_timeout 60s;
    proxy_pass http://127.0.0.1:8000;
}
```

If `MAX_UPLOAD_BYTES` changes, adjust `client_max_body_size` to be at least the
file limit plus the application's 1 MiB multipart allowance. Proxy timeout
settings also help limit slow-client resource use.

## Endpoints

### `GET /health`

Returns `200` when the API process responds:

```json
{"status":"ok"}
```

This checks API liveness; it does not indicate that an AI model is loaded.

### `POST /api/predict`

Accepts `multipart/form-data` with one field named `file`. The upload must be
a decodable image and no larger than the configured byte limit. Invalid files
return `400`, missing or malformed request fields return `422`, and oversized
files, request bodies, or image dimensions return `413`. Errors use a single
JSON shape, `{"detail":"..."}`. Unexpected server failures return a generic
`500` message; traceback and server paths are logged only on the server. For a
valid image, the current response is `503 Service Unavailable` with a clear
message that prediction is not ready. On future integration, successful
responses follow this shape:

```json
{
  "filename": "sample.png",
  "predictions": [
    {"label": "model-produced label", "confidence": 0.0}
  ]
}
```

The example describes the response schema only; the backend does not produce
example or fabricated predictions.

## Run tests

From the repository root:

```bash
python -m pytest backend/tests
```
