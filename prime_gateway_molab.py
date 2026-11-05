import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    mo.md("""
# Prime AI Gateway · Molab launcher

This is a single-file Molab notebook. It downloads the gateway source from
the GitHub repository, installs dependencies, and starts the FastAPI app
inside this notebook session.

**Important:** Molab sessions are temporary. This is for a temporary demo/test,
not a permanent public API endpoint. For production use, deploy the same repo
to a persistent FastAPI host and attach persistent storage for the SQLite DB.
""")
    return (mo,)

@app.cell
def _(mo):
    import pathlib
    import subprocess
    import sys
    import urllib.request

    repo_raw = "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/"
    workdir = pathlib.Path.cwd() / "prime-ai-gateway"
    workdir.mkdir(exist_ok=True)

    downloaded = []
    for filename in ("main.py", "requirements.txt"):
        target = workdir / filename
        urllib.request.urlretrieve(repo_raw + filename, target)
        downloaded.append(filename)

    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(workdir / "requirements.txt")],
        capture_output=True, text=True
    )
    if install.returncode != 0:
        mo.stop(True, mo.vstack([mo.md("### Dependency installation failed"), mo.code(install.stderr[-5000:])]))
    mo.md("Ready: downloaded " + ", ".join(downloaded) + " and installed dependencies.")
    return pathlib, subprocess, sys, workdir

@app.cell
def _(mo):
    admin_password = mo.ui.text(label="Admin password", kind="password")
    session_secret = mo.ui.text(label="Session secret (use a long random string)", kind="password")
    hf_token = mo.ui.text(label="Hugging Face token", kind="password")
    mo.vstack([
        mo.md("""
## قبل از اجرا: توکن Hugging Face را بساز

برای اتصال گیت‌وی به مدل‌های چت و تصویر:

1. وارد صفحهٔ رسمی [Hugging Face Tokens](https://huggingface.co/settings/tokens) شو.
2. گزینهٔ **Create new token** را بزن و نوع **Fine-grained** را انتخاب کن.
3. نامی مثل `prime-ai-gateway` برای توکن بگذار.
4. مجوز **Make calls to Inference Providers** را فعال کن و توکن را بساز.
5. توکن ساخته‌شده (معمولاً با `hf_` شروع می‌شود) را در کادر پایین وارد کن.

**تفاوت دو کلید:**
- **Hugging Face token**: دسترسی گیت‌وی به سرویس مدل‌ها.
- **Prime API key**: کلیدی که بعداً از پنل گیت‌وی می‌سازی و به برنامهٔ خودت می‌دهی.

ممکن است استفاده از مدل‌ها مشمول محدودیت مصرف یا هزینهٔ ارائه‌دهنده باشد. توکن را در کد یا GitHub قرار نده و با کسی به اشتراک نگذار.
"""),
        mo.md("## تنظیمات این نشست موقت\nبرای پنل، رمز مدیر و یک Session Secret طولانی وارد کن. این اطلاعات مخصوص همین نشست هستند."),
        admin_password, session_secret, hf_token
    ])
    return admin_password, hf_token, session_secret

@app.cell
def _(admin_password, hf_token, mo, session_secret, subprocess, sys, workdir):
    import os
    import time
    import urllib.request

    if not admin_password.value or not session_secret.value or not hf_token.value:
        mo.md("Fill all three fields above, then run this cell.")
        return

    env = os.environ.copy()
    env.update({
        "ADMIN_PASSWORD": admin_password.value,
        "SESSION_SECRET": session_secret.value,
        "HF_TOKEN": hf_token.value,
        "COOKIE_SECURE": "false",
        "DATABASE_PATH": str(workdir / "data" / "gateway.db"),
    })

    process = globals().get("_prime_gateway_process")
    if process is None or process.poll() is not None:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(workdir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        globals()["_prime_gateway_process"] = process

    time.sleep(3)
    if process.poll() is not None:
        output = process.stdout.read()[-5000:] if process.stdout else "No process output."
        mo.vstack([mo.md("### Gateway did not start"), mo.code(output)])
    else:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                health = response.read().decode()
            mo.vstack([
                mo.md("## Gateway is running in this Molab session"),
                mo.md("Health check: " + health),
                mo.md("Admin page: http://127.0.0.1:8000/"),
                mo.md("API docs: http://127.0.0.1:8000/docs"),
                mo.md("These localhost addresses are inside the notebook runtime. They are not a guaranteed public API URL. The process and SQLite database are temporary and may disappear when the session stops.")
            ])
        except Exception as exc:
            mo.md("Process started, but health check failed: " + str(exc))
    return

if __name__ == "__main__":
    app.run()
