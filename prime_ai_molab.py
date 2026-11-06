# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import importlib.util
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import marimo

__generated_with = "0.19.0"

RAW_BASE = "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main"
ASSETS = ("prime_ai_engine.py", "prime_theme.css", "pixel_ghost.html", "PRIME_AI_KNOWLEDGE.md")
PROJECT_DIR = Path(tempfile.gettempdir()) / "prime-ai-gateway"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_raw_file(filename):
    url = f"{RAW_BASE}/{filename}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Prime-AI-Molab"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        destination = PROJECT_DIR / filename
        destination.write_bytes(content)
        return destination
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Prime AI could not load {filename} from GitHub Raw: {url} ({exc})") from exc


FILES = {name: fetch_raw_file(name) for name in ASSETS}
app = marimo.App(width="full", css_file=str(FILES["prime_theme.css"]))


@app.cell
def _():
    import importlib.util
    import marimo as mo

    search_toggle = mo.ui.checkbox(
        value=False,
        label="🌐 جست‌وجوی وب (استفاده از نتایج اینترنت در پاسخ‌ها)",
    )

    engine_path = FILES["prime_ai_engine.py"]
    spec = importlib.util.spec_from_file_location("prime_ai_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import loader for {engine_path}")

    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    prime_ui = engine.build_prime_ai(search_toggle)
    prime_ui
    return (prime_ui,)

if __name__ == "__main__":
    app.run()
