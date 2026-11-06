# Prime AI

Prime AI loads its project files from individual GitHub Raw URLs at startup. The Molab entry point uses one Marimo cell.

## Molab entry point

Open `prime_ai_molab.py` in Molab.

## Project files

- `prime_ai_molab.py` / `run.py` — one-cell Molab loader.
- `prime_ai_engine.py` — chat, model routing, Qwen-Image generation, and UI.
- `prime_theme.css` — dark industrial theme.
- `pixel_ghost.html` — transparent floating pixel animation.
- `PRIME_AI_KNOWLEDGE.md` — editable project knowledge.
- `main.py` — separate FastAPI gateway service.

At startup, the loader downloads each asset from its own explicit Raw URL into a temporary directory. Chat and image models load on demand. Qwen-Image is the only image-generation model.

Enable a GPU in Molab before sending chat or image requests. The first request may take time while model weights are downloaded.

## Custom chat UI

The notebook uses a standalone AnyWidget interface instead of Marimo's built-in chat component. The component is split across `prime_chat_widget.py`, `prime_chat.js`, and `prime_chat.css`. The loader fetches these files from GitHub Raw, and the Python engine connects the widget request/response traits to the model and routing functions. The interface includes a conversation view, message composer, file upload, Markdown/code display, generated image display, and new-chat action. A live Molab runtime test is still required.

## Image generation

When Prime AI identifies an image-generation request, it automatically downloads and uses **Qwen-Image**. The chat status displays download progress and the current loading stage. No image-model selection step is shown, and only one image pipeline is kept active at a time to reduce GPU memory use.

The image-generation flow has not yet been tested in a live Molab runtime.
