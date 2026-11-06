# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import importlib.util
    import tempfile
    import urllib.error
    import urllib.request
    from pathlib import Path

    import marimo as mo

    # Each project file has its own direct GitHub Raw URL.
    RAW_FILES = {
        "prime_ai_engine.py": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/prime_ai_engine.py",
        "prime_theme.css": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/prime_theme.css",
        "pixel_ghost.html": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/pixel_ghost.html",
        "PRIME_AI_KNOWLEDGE.md": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/PRIME_AI_KNOWLEDGE.md",
    }

    project_dir = Path(tempfile.gettempdir()) / "prime-ai-gateway"
    project_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    for filename, url in RAW_FILES.items():
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Prime-AI-Molab"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            destination = project_dir / filename
            destination.write_bytes(content)
            files[filename] = destination
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Prime AI could not load {filename} from GitHub Raw: {url} ({exc})"
            ) from exc

    search_toggle = mo.ui.checkbox(
        value=False,
        label="🌐 جست‌وجوی وب (استفاده از نتایج اینترنت در پاسخ‌ها)",
    )

    spec = importlib.util.spec_from_file_location(
        "prime_ai_engine", files["prime_ai_engine.py"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create an import loader for {files['prime_ai_engine.py']}"
        )

    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    prime_ui = engine.build_prime_ai(search_toggle)

    # Apply the stylesheet fetched from its own Raw URL.
    css_text = files["prime_theme.css"].read_text(encoding="utf-8")
    style = mo.Html(f"<style>{css_text}</style>")

    mo.vstack([style, prime_ui])
    return


if __name__ == "__main__":
    app.run()
