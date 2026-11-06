# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
import marimo
from pathlib import Path

__generated_with = "0.19.0"
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_CSS_PATH = _SCRIPT_DIR / "prime_theme.css"
app = marimo.App(width="full", css_file=str(_CSS_PATH) if _CSS_PATH.is_file() else None)


@app.cell
def _():
    import marimo as mo

    search_toggle = mo.ui.checkbox(
        value=False,
        label="🌐 جست‌وجوی وب (استفاده از نتایج اینترنت در پاسخ‌ها)",
    )
    return (search_toggle,)


@app.cell
def _(search_toggle):
    import marimo as mo
    from prime_ai_engine import build_prime_ai

    prime_ui = build_prime_ai(search_toggle)
    prime_ui
    return (prime_ui,)


if __name__ == "__main__":
    app.run()
