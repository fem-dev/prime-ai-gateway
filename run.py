# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
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
        "prime_chat_widget.py": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/prime_chat_widget.py",
        "prime_chat.js": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/prime_chat.js",
        "prime_chat.css": "https://raw.githubusercontent.com/fem-dev/prime-ai-gateway/main/prime_chat.css",
    }

    project_dir = Path(tempfile.gettempdir()) / "prime-ai-gateway"
    project_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    for filename, url in RAW_FILES.items():
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Prime-AI-Molab", "Cache-Control": "no-cache", "Pragma": "no-cache"}
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

    # Create the Anima prompt locally. Keep user-added/private text out of GitHub.
    private_prompt = project_dir / "ANIMA_STYLE_PROMPT.md"
    base_prompt = "# راهنمای ساخت تصویر دقیق\n\nتصویر را دقیقاً بر اساس درخواست کاربر بساز.\n\n- موضوع اصلی و منظور دقیق کاربر را حفظ کن.\n- تمام جزئیاتی را که کاربر گفته، از جمله سوژه، مکان، حالت، تعداد، جایگاه عناصر، زاویهٔ دید، سبک و ترکیب‌بندی رعایت کن.\n- موضوع درخواست را با موضوعی دیگر جایگزین نکن و آن را به برداشت کلی یا معمولی تبدیل نکن.\n- چیزی را که کاربر درخواست نکرده، به تصویر اضافه نکن؛ از عناصر بی‌ربط، نوشته، لوگو، شخصیت یا اشیای اضافی خودداری کن.\n- اگر کاربر صحنه‌ای خیالی یا ناممکن توصیف کرد، همان صحنهٔ خیالی را مطابق توصیفش تصویرسازی کن.\n- عناصر مهم درخواست را حذف یا جابه‌جا نکن.\n- ترکیب‌بندی را روشن و منسجم نگه دار تا ایدهٔ اصلی در نگاه اول قابل تشخیص باشد.\n- پیش از ساخت تصویر، جزئیات درخواست را با هم تطبیق بده و مطمئن شو خروجی از موضوع اصلی منحرف نمی‌شود.\n- نتیجه باید بازنمایی وفادارانهٔ خواستهٔ کاربر باشد، نه صرفاً تصویری که چند واژهٔ درخواست را تداعی می‌کند.\n\n<!-- متن اختصاصی خودت را از خط بعد اضافه کن. -->\n"
    old_template = "# Anima style / inspiration prompt\\n\\n<!-- Paste your private style prompt below this line. -->\\n"
    if not private_prompt.exists() or private_prompt.read_text(encoding="utf-8") == old_template:
        private_prompt.write_text(base_prompt, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "prime_ai_engine", files["prime_ai_engine.py"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create an import loader for {files['prime_ai_engine.py']}"
        )

    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    prime_ui = engine.build_prime_ai()

    # Poll the GitHub main-branch head every 5 seconds. When it changes,
    # reload the notebook so the one-cell loader fetches all fresh Raw files.
    import json

    initial_head_request = urllib.request.Request(
        "https://api.github.com/repos/fem-dev/prime-ai-gateway/commits/main",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Prime-AI-Molab"},
    )
    with urllib.request.urlopen(initial_head_request, timeout=20) as response:
        initial_head = json.loads(response.read().decode("utf-8"))["sha"]
        initial_etag = response.headers.get("ETag", "")

    updater = mo.Html(f"""
    <div id="prime-update-status" role="status" aria-live="polite"
         style="position:fixed;top:12px;right:12px;z-index:999999;display:none;
                padding:8px 14px;border:1px solid #7f1d1d;border-radius:8px;
                background:#090909;color:#ff5555;font:600 13px monospace;">
      updating....
    </div>
    <script>
    (() => {{
      if (window.__primeGitHubWatcherStarted) return;
      window.__primeGitHubWatcherStarted = true;
      const knownSha = {json.dumps(initial_head)};
      const knownEtag = {json.dumps(initial_etag)};
      const status = document.getElementById('prime-update-status');
      let checking = false;
      async function checkForUpdate() {{
        if (checking || document.hidden) return;
        checking = true;
        try {{
          const response = await fetch(
            'https://api.github.com/repos/fem-dev/prime-ai-gateway/commits/main',
            {{cache:'no-store', headers:{{'Accept':'application/vnd.github+json', 'If-None-Match':knownEtag}}}}
          );
          if (response.status === 304) return;
          if (!response.ok) throw new Error('GitHub API ' + response.status);
          const data = await response.json();
          if (data.sha && data.sha !== knownSha) {{
            status.style.display = 'block';
            setTimeout(() => window.location.reload(), 1200);
          }}
        }} catch (error) {{
          console.warn('Prime AI update check failed:', error);
        }} finally {{ checking = false; }}
      }}
      setInterval(checkForUpdate, 5000);
    }})();
    </script>
    """)

    # Apply the stylesheet fetched from its own Raw URL.
    css_text = files["prime_theme.css"].read_text(encoding="utf-8")
    style = mo.Html(f"<style>{css_text}</style>")

    mo.vstack([updater, style, prime_ui])
    return


if __name__ == "__main__":
    app.run()
