# Prime AI Gateway

A self-hosted API gateway with a password-protected admin dashboard, generated `prime-` API keys, and OpenAI-style routes for Qwen3 chat and FLUX.1 Schnell image generation.

## Features

- Admin login and dashboard.
- Generate gateway API keys and download a TXT file containing:
  - `api key : prime-...`
  - `base url : https://your-host/v1`
- Only SHA-256 hashes of gateway keys are stored. The full key is shown once.
- `POST /v1/chat/completions` routes requests to Qwen3.
- `POST /v1/images/generations` routes prompts to FLUX.1 Schnell.
- `GET /v1/models` lists the supported gateway model names.
- `GET /health` is a health check.

## Important security note

A `prime-...` key authenticates with this gateway; it is not a provider key. The server separately needs `HF_TOKEN` with permission to use the configured models. Keep that token on the server and never put it in the browser, this repository, or the downloaded TXT file.

The dashboard password is configured with `ADMIN_PASSWORD`, not hardcoded in source code. Use a long unique password and a random `SESSION_SECRET`.

## Run locally

1. Use Python 3.11 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Set `ADMIN_PASSWORD`, `SESSION_SECRET`, and `HF_TOKEN` in your shell or hosting provider's environment settings. The app does not automatically load `.env`.
5. Run: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Open the server root URL, sign in, and generate a key.

For local testing set `PUBLIC_BASE_URL=http://localhost:8000`. For public deployment set it to the HTTPS origin assigned to the app.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `ADMIN_PASSWORD` | Yes | Dashboard password |
| `SESSION_SECRET` | Yes | Signs the admin session cookie |
| `HF_TOKEN` | Yes for model calls | Server-side provider credential |
| `PUBLIC_BASE_URL` | Recommended | Public origin used in downloaded TXT files |
| `QWEN_MODEL` | No | Qwen provider model identifier |
| `QWEN_API_URL` | No | OpenAI-compatible chat endpoint |
| `FLUX_API_URL` | No | FLUX inference endpoint |
| `COOKIE_SECURE` | No | Set to `true` behind HTTPS |
| `DATABASE_PATH` | No | SQLite file path; use persistent storage |

## API examples

Replace the example key and host with the values in your downloaded TXT file.

### Chat completion

```bash
curl https://your-host/v1/chat/completions \
  -H "Authorization: Bearer prime-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}]}'
```

### Image generation

```bash
curl https://your-host/v1/images/generations \
  -H "Authorization: Bearer prime-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux.1-schnell","prompt":"A watercolor illustration of a small lighthouse"}'
```

The image response contains a base64-encoded image in `data[0].b64_json`.

### List models

```bash
curl https://your-host/v1/models -H "Authorization: Bearer prime-YOUR_KEY"
```

## Deployment

This is a Python ASGI web app. The host must support a long-running web process and outbound HTTPS requests to the configured provider. Set environment variables in the host's secret/configuration settings.

Attach persistent storage and point `DATABASE_PATH` to it, otherwise API-key hashes may be lost when the service is rebuilt or restarted. Set `COOKIE_SECURE=true` for HTTPS. A notebook runtime that shuts down when its session ends is not a reliable always-on public API.

## Current limitations

- Chat streaming (`stream=true`) is not implemented; use `stream=false`.
- Image endpoint currently accepts one image per request.
- Provider model access, quotas, and endpoint compatibility depend on the provider account and deployment configuration.
- This starter does not yet include usage metering, per-key quotas, or a key-revocation UI (revocation endpoint is available to an authenticated admin).

## Security checklist

- Use unique strong admin and session secrets.
- Never commit `.env`, provider tokens, API keys, or the SQLite database.
- Treat downloaded gateway keys like passwords.
- Use HTTPS for public traffic.
- Revoke a gateway key immediately if it is exposed.
