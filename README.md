# Prime AI

Prime AI loads its application files directly from GitHub Raw at startup.

## Molab entry point

Open `prime_ai_molab.py` in Molab. The notebook downloads the latest files from the `main` branch using this base URL:

`https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/`

## Project files

- `prime_ai_molab.py` — compact Molab entry point and Raw loader.
- `prime_ai_engine.py` — model loading, chat routing, web search, and image generation.
- `pixel_ghost.html` — floating Pixel Ghost animation.
- `prime_theme.css` — Prime AI theme.
- `PRIME_AI_KNOWLEDGE.md` — editable project knowledge.
- `main.py` — separate FastAPI gateway service.
- `requirements.txt` — gateway dependencies.

At startup, the notebook downloads the engine, CSS, Pixel Ghost HTML, and knowledge file from GitHub Raw into a temporary project directory. It then imports the downloaded engine and uses the downloaded CSS and assets. A failed download raises an error naming the exact file and Raw URL, so GitHub/network loading problems are easier to identify.

The first launch may take longer while Python dependencies and model weights are installed. Enable a GPU in Molab for local inference.
