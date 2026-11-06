# Prime AI

Prime AI's Molab notebook is split into small, version-controlled files so the notebook itself stays readable.

## Molab entry point

Open `prime_ai_molab.py` from this GitHub repository in Molab. Molab makes the other files in the repository available beside the notebook when it is opened from GitHub.

## Project layout

- `prime_ai_molab.py` — small Marimo entry point and UI cells.
- `prime_ai_engine.py` — model loading, chat routing, web search, and image generation.
- `pixel_ghost.html` — floating Pixel Ghost animation.
- `prime_theme.css` — Prime AI dark theme.
- `PRIME_AI_KNOWLEDGE.md` — editable project knowledge and instructions.
- `main.py` — separate FastAPI API-gateway service; it is not the Molab notebook.
- `requirements.txt` — dependencies for the FastAPI gateway.

The notebook imports `build_prime_ai` from `prime_ai_engine.py`; it reads the HTML, CSS, and knowledge file from the same repository directory. Keep these files together when opening the notebook in Molab.

## Runtime notes

The Molab app needs a GPU enabled for local Qwen and Anima inference. The first run may install Python packages and download model weights, so initial startup can take longer than later launches.

For Molab troubleshooting, check the app's startup logs if launch fails; a generic launch banner alone does not identify the underlying exception.
