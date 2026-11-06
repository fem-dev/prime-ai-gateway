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

        # Add a separate viewer link and a native PNG download control.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode("ascii")

        mo.output.append(
            mo.vstack([
                mo.md("### تصویرت آماده شد!"),
                mo.Html(
                    '<a href="data:image/png;base64,' + encoded +
                    '" target="_blank" rel="noopener noreferrer" '
                    'style="display:inline-block;padding:10px 16px;'
                    'border-radius:9px;background:var(--marimo-foreground);'
                    'color:var(--marimo-background);text-decoration:none;'
                    'font-weight:600;">نمایش تصویر در تب جدید ↗</a>'
                ),
                mo.download(
                    data=image_bytes,
                    filename="prime-ai-image.png",
                    mimetype="image/png",
                    label="دانلود تصویر PNG",
                ),
            ])
        )
        return "تصویرت آماده شد! برای نمایش یا دانلود، گزینه‌های پایین خروجی را انتخاب کن."

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

    mo.vstack([
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
