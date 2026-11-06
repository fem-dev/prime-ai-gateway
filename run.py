import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Prime AI: one-cell, single-chat interface with automatic image routing.
    import base64
    import gc
    import io
    import json
    import re
    import subprocess
    import sys
    import threading
    from pathlib import Path

    import marimo as mo

    INSTALL_MARKER = Path.home() / ".prime_ai_deps_installed_search_anima_v2"
    if not INSTALL_MARKER.exists():
        packages = [
            "transformers>=4.51.0",
            "git+https://github.com/huggingface/diffusers",
            "accelerate>=1.6.0",
            "safetensors>=0.5.3",
            "sentencepiece>=0.2.0",
            "ddgs>=9.0.0",
        ]
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            mo.stop(
                True,
                mo.vstack([
                    mo.md("## نصب وابستگی‌ها ناموفق بود"),
                    mo.code((result.stderr or result.stdout)[-5000:]),
                ]),
            )
        INSTALL_MARKER.touch()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    QWEN_ID = "Qwen/Qwen3-32B"
    IMAGE_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
    state = globals().setdefault(
        "_prime_ai_state",
        {
            "qwen": None,
            "qwen_tokenizer": None,
            "image_pipe": None,
            "image_loading": False,
            "image_error": None,
            "lock": threading.RLock(),
        },
    )

    def unload_image_model():
        if state["image_pipe"] is not None:
            del state["image_pipe"]
            state["image_pipe"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_chat_model():
        if state["qwen"] is not None:
            del state["qwen"]
            state["qwen"] = None
            state["qwen_tokenizer"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_qwen():
        with state["lock"]:
            # Only one large model stays on the GPU at a time.
            unload_image_model()
            if state["qwen"] is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("برای اجرای مدل‌ها، GPU را در تنظیمات Molab فعال کن.")
                tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    QWEN_ID,
                    device_map="auto",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                state["qwen_tokenizer"] = tokenizer
                state["qwen"] = model
            return state["qwen"], state["qwen_tokenizer"]

    KNOWLEDGE_FILE = Path.cwd() / "PRIME_AI_KNOWLEDGE.md"

    def load_project_knowledge():
        try:
            if KNOWLEDGE_FILE.exists():
                return KNOWLEDGE_FILE.read_text(encoding="utf-8")[:18000]
        except Exception as exc:
            return f"Knowledge file could not be read: {exc}"
        return "No additional project knowledge file was found."

    def search_web(query, max_results=5):
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        lines = []
        for item in results or []:
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            href = str(item.get("href", "")).strip()
            if title or body:
                lines.append(f"- {title}\\n  {body}\\n  Source: {href}")
        return "\\n\\n".join(lines) if lines else "No useful web results were found."

    def ask_qwen(messages):
        model, tokenizer = load_qwen()
        conversation = []
        for item in messages:
            role = getattr(item, "role", "user")
            text = getattr(item, "content", "")
            if isinstance(text, str) and role in ("user", "assistant", "system"):
                conversation.append({"role": role, "content": text})
        if not conversation:
            return "لطفاً یک پیام بفرست."

        # Qwen decides whether this turn needs an image or a normal text reply.
        router_instruction = {
            "role": "system",
            "content": (
                "You are Prime AI's request router and assistant. Read the conversation. "
                "If the latest user message clearly asks to create, generate, draw, or make "
                "an image/picture/illustration, return ONLY valid JSON in this exact shape: "
                '{"action":"image","prompt":"a detailed image-generation prompt"} . '
                "Translate and expand the user's image idea into a clear visual prompt. "
                "Otherwise return ONLY valid JSON in this exact shape: "
                '{"action":"chat","response":"your helpful answer in the user\'s language"}. '
                "Do not route to image generation when the user is merely discussing images, "
                "asking how to make them, or asking a question about an existing image. "
                "Use the project knowledge supplied below when relevant. "
                "Project knowledge: " + load_project_knowledge()
            ),
        }
        routed = [router_instruction] + conversation
        prompt = tokenizer.apply_chat_template(
            routed,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with state["lock"], torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {"action": "chat", "response": raw or "متوجه درخواستت نشدم؛ دوباره بگو."}
        try:
            result = json.loads(match.group(0))
            if result.get("action") == "image" and result.get("prompt"):
                return {"action": "image", "prompt": str(result["prompt"])}
            if result.get("action") == "chat":
                return {"action": "chat", "response": str(result.get("response", ""))}
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return {"action": "chat", "response": raw}

    def ensure_image_model():
        with state["lock"]:
            if state["image_pipe"] is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("برای بارگذاری مدل تصویر، GPU را در تنظیمات Molab فعال کن.")
                from diffusers import ModularPipeline
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                state["image_loading"] = True
                state["image_error"] = None
                try:
                    # Official Anima Diffusers distribution uses the modular pipeline.
                    pipe = ModularPipeline.from_pretrained(IMAGE_ID)
                    pipe.load_components(dtype=dtype)
                    state["image_pipe"] = pipe.to("cuda")
                except Exception as exc:
                    state["image_error"] = str(exc)
                    raise
                finally:
                    state["image_loading"] = False
            return state["image_pipe"]

    def preload_image_model():
        try:
            ensure_image_model()
            print("Prime AI: Anima آماده است.")
        except Exception as exc:
            print("Prime AI: بارگذاری اولیه Z-Anime ناموفق بود:", str(exc)[-2000:])

    def generate_image(prompt_text):
        if not torch.cuda.is_available():
            return "برای ساخت تصویر باید GPU را در تنظیمات Molab فعال کنی."

        with state["lock"]:
            unload_chat_model()
            pipe = ensure_image_model()
            result = pipe(
                prompt="masterpiece, best quality, safe, " + prompt_text,
                width=832,
                height=1216,
                num_inference_steps=40,
                guidance_scale=4.5,
                negative_prompt="worst quality, low quality, blurry, jpeg artifacts, chromatic aberration, bad anatomy, extra fingers, missing fingers, deformed hands, text, watermark",
                generator=torch.Generator(device="cuda").manual_seed(
                    int(torch.randint(0, 2**31 - 1, (1,)).item())
                ),
            )
            image = result.images[0]

        # Return the image as the chat response itself. Appending to the notebook
        # output from inside a chat callback can disrupt subsequent chat turns.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return mo.vstack([
            mo.md("### تصویرت آماده شد!"),
            mo.Html(
                '<div style="max-width:560px;margin:8px 0;">'
                '<img alt="Prime AI generated image" '
                'src="data:image/png;base64,' + encoded + '" '
                'style="display:block;width:100%;height:auto;border-radius:16px;'
                'border:1px solid #34343d;box-shadow:0 12px 40px #0008;">'
                '</div>'
            ),
            mo.download(
                data=image_bytes,
                filename="prime-ai-image.png",
                mimetype="image/png",
                label="دانلود تصویر PNG",
            ),
        ])

    def answer(messages, config):
        try:
            decision = ask_qwen(messages)
            if decision.get("action") == "image":
                return generate_image(decision["prompt"])

            if search_toggle.value:
                last_user = next(
                    (getattr(item, "content", "") for item in reversed(messages)
                     if getattr(item, "role", "user") == "user"),
                    "",
                )
                if last_user.strip():
                    web_results = search_web(last_user.strip())
                    enriched_messages = list(messages) + [{
                        "role": "system",
                        "content": (
                            "Use these web search results to answer the latest question. "
                            "Distinguish facts from uncertainty, include source URLs as plain text, "
                            "and never invent sources.\\n\\n" + web_results
                        ),
                    }]
                    decision = ask_qwen(enriched_messages)
            return decision.get("response", "متوجه درخواستت نشدم.")
        except Exception as exc:
            return "خطا در اجرای Prime AI:\n\n" + str(exc)[-2500:]

    # Start downloading and importing the anime model as soon as this app cell runs.
    preload_thread = threading.Thread(target=preload_image_model, daemon=True)
    preload_thread.start()

    search_toggle = mo.ui.checkbox(
        value=False,
        label="🌐 جست‌وجوی وب (استفاده از نتایج اینترنت در پاسخ‌ها)",
    )

    chat = mo.ui.chat(
        answer,
        prompts=[
            "سلام! خودت را معرفی کن.",
            "یک ایده برای پروژه برنامه‌نویسی بده.",
            "یک تصویر فانتزی از شهری روی ابرها بساز.",
        ],
        show_configuration_controls=False,
    )

    fullscreen_button = mo.Html("""
<button id="prime-fullscreen-btn" type="button"
style="float:right;padding:9px 14px;border:1px solid var(--border-color);
border-radius:10px;background:var(--surface);color:var(--foreground);
cursor:pointer;font-weight:600;">
⛶ نمایش چت تمام‌صفحه
</button>
<script>
(() => {
  const btn = document.getElementById("prime-fullscreen-btn");
  if (!btn || btn.dataset.ready) return;
  btn.dataset.ready = "1";
  btn.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        const target = document.querySelector("main") || document.documentElement;
        await target.requestFullscreen();
      }
    } catch (err) {
      btn.textContent = "تمام‌صفحه در این مرورگر پشتیبانی نمی‌شود";
    }
  });
  document.addEventListener("fullscreenchange", () => {
    btn.textContent = document.fullscreenElement
      ? "⤢ خروج از تمام‌صفحه"
      : "⛶ نمایش چت تمام‌صفحه";
  });
})();
</script>
""")

    prime_theme = mo.Html("""
<style>
:root { color-scheme: dark; }
html, body { background:#090a0d !important; color:#e9e8e5 !important; }
body { background:radial-gradient(ellipse at 50% 0%,#171923 0%,#090a0d 58%) fixed !important; }
main { max-width:1180px !important; margin-left:250px !important; margin-right:230px !important; }
main, [data-testid*="chat"], [class*="chat"] { color:#e9e8e5; }
.prime-sidebar {
 position:fixed; z-index:999; inset:0 auto 0 0; width:226px; padding:22px 15px;
 background:rgba(12,13,17,.96); border-right:1px solid #252630;
 display:flex; flex-direction:column; gap:18px; backdrop-filter:blur(18px);
 font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; color:#dedee6;
}
.prime-brand {display:flex;align-items:center;gap:10px;font-weight:750;font-size:17px;letter-spacing:.12em}
.prime-brand-mark {width:30px;height:30px;border-radius:10px;display:grid;place-items:center;
 background:linear-gradient(135deg,#a78bfa,#6d5dfc);color:white;box-shadow:0 0 22px #8b5cf655}
.prime-new {border:1px solid #383946;border-radius:11px;padding:11px 12px;background:#191a22;color:#f4f3f8;text-align:left}
.prime-new:hover {border-color:#8b7cff;background:#20212c}
.prime-nav-label {color:#777987;font-size:11px;letter-spacing:.13em;text-transform:uppercase;padding:3px 8px}
.prime-nav-item {padding:9px 10px;border-radius:9px;color:#aeb0bc}
.prime-nav-item.active {background:#20212b;color:#f4f2ff}
.prime-sidebar-foot {margin-top:auto;border-top:1px solid #282a34;padding:14px 6px;color:#8c8e9b;font-size:12px}
.prime-mascot {
 position:fixed;z-index:998;right:18px;bottom:18px;width:190px;padding:12px 10px 10px;
 border:1px solid #2d2d3b;border-radius:17px;background:linear-gradient(160deg,#171823ee,#0d0e14f2);
 box-shadow:0 18px 60px #0009;backdrop-filter:blur(16px);text-align:center;
 color:#b9bac8;font:11px/1.5 ui-monospace,SFMono-Regular,monospace;
}
.prime-mascot-head {display:flex;justify-content:space-between;align-items:center;margin:0 2px 5px}
.prime-live {color:#8fe2b2}.prime-live:before {content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:#66e59c;margin-right:6px;box-shadow:0 0 9px #66e59c}
.prime-pixel {width:160px;height:155px;image-rendering:pixelated;shape-rendering:crispEdges;overflow:visible}
.prime-arm-left {transform-origin:66px 103px;animation:prime-type-a .34s steps(2,end) infinite}
.prime-arm-right {transform-origin:103px 103px;animation:prime-type-b .29s steps(2,end) infinite}
.prime-eye {transform-origin:center;animation:prime-blink 4.2s steps(1,end) infinite}
.prime-screen {animation:prime-screen 2.4s ease-in-out infinite}
.prime-spark {animation:prime-float 2.8s ease-in-out infinite}
@keyframes prime-type-a {0%,100%{transform:rotate(-3deg)}50%{transform:rotate(9deg) translate(2px,2px)}}
@keyframes prime-type-b {0%,100%{transform:rotate(7deg)}50%{transform:rotate(-8deg) translate(-2px,1px)}}
@keyframes prime-blink {0%,94%,100%{transform:scaleY(1)}95%,97%{transform:scaleY(.12)}}
@keyframes prime-screen {0%,100%{opacity:.65}50%{opacity:1}}
@keyframes prime-float {0%,100%{transform:translateY(0);opacity:.45}50%{transform:translateY(-5px);opacity:1}}
textarea,input,[contenteditable="true"] {background:#15161d !important;color:#f0eff7 !important;border-color:#353642 !important;border-radius:13px !important}
button {transition:background .18s ease,border-color .18s ease,transform .18s ease}
button:hover {border-color:#8274f5 !important}
@media(max-width:1050px) {main{margin-left:220px !important;margin-right:18px !important}.prime-mascot{display:none}}
@media(max-width:720px) {.prime-sidebar{display:none}main{margin:0 10px !important;padding:8px !important}}
@media(prefers-reduced-motion:reduce) {.prime-arm-left,.prime-arm-right,.prime-eye,.prime-screen,.prime-spark{animation:none!important}}
</style>
<aside class="prime-sidebar">
 <div class="prime-brand"><span class="prime-brand-mark">✦</span><span>PRIME AI</span></div>
 <button class="prime-new" type="button">＋　گفت‌وگوی جدید</button>
 <div class="prime-nav-label">Workspace</div>
 <div class="prime-nav-item active">◈　چت هوشمند</div>
 <div class="prime-nav-item">⌕　جست‌وجوی وب</div>
 <div class="prime-nav-item">▧　ساخت تصویر</div>
 <div class="prime-nav-label">Recent</div>
 <div class="prime-nav-item">✧　ایده‌های خلاقانه</div>
 <div class="prime-nav-item">⌘　پروژه‌های کدنویسی</div>
 <div class="prime-nav-item">▤　پرامپت‌های من</div>
 <div class="prime-sidebar-foot">● سیستم آماده است<br><span style="color:#686b79">Qwen3 · Anima · Web Search</span></div>
</aside>
<aside class="prime-mascot">
 <div class="prime-mascot-head"><span>PIXEL ASSISTANT</span><span class="prime-live">ONLINE</span></div>
 <svg class="prime-pixel" viewBox="0 0 160 155" role="img" aria-label="دختر انیمه‌ای پیکسلی در حال تایپ روی لپ‌تاپ">
  <g class="prime-spark" fill="#9d8cff"><rect x="17" y="30" width="4" height="4"/><rect x="137" y="22" width="5" height="5"/><rect x="126" y="68" width="3" height="3"/></g>
  <rect x="40" y="131" width="82" height="5" fill="#343447"/>
  <rect x="51" y="48" width="57" height="54" fill="#38264b"/>
  <rect x="45" y="57" width="69" height="42" fill="#6d477e"/>
  <rect x="49" y="53" width="62" height="43" fill="#f2c7b5"/>
  <rect x="43" y="34" width="72" height="36" fill="#29213e"/>
  <rect x="37" y="43" width="15" height="42" fill="#29213e"/>
  <rect x="105" y="40" width="16" height="45" fill="#29213e"/>
  <rect x="47" y="29" width="60" height="18" fill="#29213e"/>
  <rect x="54" y="20" width="46" height="15" fill="#29213e"/>
  <rect x="62" y="16" width="29" height="8" fill="#29213e"/>
  <rect x="47" y="48" width="10" height="20" fill="#29213e"/>
  <rect x="103" y="47" width="9" height="23" fill="#29213e"/>
  <g fill="#f7d5c5"><rect x="58" y="49" width="43" height="36"/><rect x="51" y="60" width="8" height="15"/><rect x="100" y="60" width="8" height="15"/></g>
  <g fill="#6a4162"><rect x="66" y="62" width="8" height="4"/><rect x="86" y="62" width="8" height="4"/><rect x="76" y="77" width="10" height="3"/></g>
  <g class="prime-eye" fill="#342344"><rect x="67" y="64" width="5" height="7"/><rect x="87" y="64" width="5" height="7"/></g>
  <rect x="75" y="72" width="5" height="3" fill="#e58c91"/>
  <rect x="55" y="84" width="50" height="31" fill="#7463c7"/>
  <rect x="64" y="89" width="33" height="8" fill="#a99bf0"/>
  <g class="prime-arm-left"><rect x="52" y="91" width="13" height="24" fill="#f2c7b5"/><rect x="57" y="109" width="13" height="7" fill="#f2c7b5"/></g>
  <g class="prime-arm-right"><rect x="97" y="91" width="13" height="24" fill="#f2c7b5"/><rect x="99" y="109" width="13" height="7" fill="#f2c7b5"/></g>
  <rect x="27" y="111" width="106" height="7" fill="#77788f"/>
  <rect x="35" y="118" width="90" height="25" fill="#252a3d"/>
  <rect x="41" y="123" width="78" height="14" fill="#414a68"/>
  <g class="prime-screen" fill="#a5a0ff"><rect x="47" y="126" width="12" height="2"/><rect x="64" y="126" width="23" height="2"/><rect x="92" y="126" width="18" height="2"/><rect x="52" y="132" width="26" height="2"/><rect x="84" y="132" width="17" height="2"/></g>
  <rect x="71" y="143" width="18" height="3" fill="#626985"/>
 </svg>
 <div>typing code<span class="prime-spark"> ▋</span></div>
</aside>
""")

    mo.vstack([
        prime_theme,
        fullscreen_button,
        mo.md("""
# ✦ PRIME AI
### یک چت هوشمند برای گفت‌وگو و ساخت تصویر

همه‌چیز از همین کادر چت انجام می‌شود:
- **Qwen3-32B (BF16)** برای گفت‌وگو و تشخیص درخواست ساخت تصویر
- **Anima 2B (BF16)** برای تولید تخصصی تصویر انیمه
- **جست‌وجوی وب** با فعال‌کردن گزینه مربوط، برای افزودن نتایج اینترنت به پاسخ‌ها
- **فایل دانش پروژه** از PRIME_AI_KNOWLEDGE.md برای اطلاعات و دستورالعمل‌های اختصاصی

فقط درخواست ساخت تصویرت را در چت بنویس؛ Prime AI تشخیص می‌دهد و تصویر را همان‌جا در گفت‌وگو نمایش می‌دهد.
برای اجرای مدل‌ها GPU را در تنظیمات Molab فعال کن. بار اول، وزن‌های عمومی مدل‌ها بدون API Key یا توکن Hugging Face دانلود می‌شوند. فایل PRIME_AI_KNOWLEDGE.md را کنار فایل برنامه نگه دار و برای افزودن اطلاعات اختصاصی ویرایش کن.
"""),
        search_toggle,
        chat,
        mo.md("---\nPrime AI · یک سلول، یک چت، دو مدل محلی."),
    ])


if __name__ == "__main__":
    app.run()
