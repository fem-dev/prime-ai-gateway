# Prime AI

Prime AI loads its project files from individual GitHub Raw URLs at startup. The Molab entry point uses one Marimo cell.

## Molab entry point

Open `prime_ai_molab.py` in Molab.

## Project files

- `prime_ai_molab.py` / `run.py` — one-cell Molab loader.
- `prime_ai_engine.py` — chat, model routing, image generation, and UI.
- `prime_theme.css` — dark industrial theme.
- `pixel_ghost.html` — transparent floating pixel animation.
- `PRIME_AI_KNOWLEDGE.md` — editable project knowledge.
- `main.py` — separate FastAPI gateway service.

At startup, the loader downloads each asset from its own explicit Raw URL into a temporary directory. The image model is no longer preloaded at notebook startup; Qwen loads when the first chat request arrives, and Anima loads only when an image is requested. This avoids a large background GPU allocation while Molab is trying to start the kernel.

Enable a GPU in Molab before sending chat or image requests. The first request may take time while model weights are downloaded.


## Custom chat UI

The notebook now uses a standalone AnyWidget interface instead of Marimo's built-in chat component. The component is split across `prime_chat_widget.py`, `prime_chat.js`, and `prime_chat.css`. The loader fetches these files from GitHub Raw, and the Python engine connects the widget request/response traits to the existing model and routing functions. The custom interface includes a conversation view, message composer, file upload, Markdown/code display, generated image display, and new-chat action. A live Molab runtime test is still required.
