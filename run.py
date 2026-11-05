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

    INSTALL_MARKER = Path.home() / ".prime_ai_deps_installed_qwen_image_2512"
    if not INSTALL_MARKER.exists():
        packages = [
            "transformers>=4.51.0",
            "git+https://github.com/huggingface/diffusers",
            "accelerate>=1.6.0",
            "safetensors>=0.5.3",
            "sentencepiece>=0.2.0",
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
    IMAGE_ID = "Qwen/Qwen-Image-2512"
    state = globals().setdefault(
        "_prime_ai_state",
        {
            "qwen": None,
            "qwen_tokenizer": None,
            "image_pipe": None,
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
                "asking how to make them, or asking a question about an existing image."
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

    def generate_image(prompt_text):
        if not torch.cuda.is_available():
            return "برای ساخت تصویر باید GPU را در تنظیمات Molab فعال کنی."
        from diffusers import DiffusionPipeline

        with state["lock"]:
            unload_chat_model()
            if state["image_pipe"] is None:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                state["image_pipe"] = DiffusionPipeline.from_pretrained(
                    IMAGE_ID,
                    torch_dtype=dtype,
                ).to("cuda")
            result = state["image_pipe"](
                prompt=prompt_text,
                width=1328,
                height=1328,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(
                    int(torch.randint(0, 2**31 - 1, (1,)).item())
                ),
            )
            image = result.images[0]

        # Return the generated image inline in the same chat conversation.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "تصویرت آماده شد!\n\n![تصویر تولیدشده](data:image/png;base64," + encoded + ")"

    def answer(messages, config):
        try:
            decision = ask_qwen(messages)
            if decision.get("action") == "image":
                return generate_image(decision["prompt"])
            return decision.get("response", "متوجه درخواستت نشدم.")
        except Exception as exc:
            return "خطا در اجرای Prime AI:\n\n" + str(exc)[-2500:]

    chat = mo.ui.chat(
        answer,
        prompts=[
            "سلام! خودت را معرفی کن.",
            "یک ایده برای پروژه برنامه‌نویسی بده.",
            "یک تصویر فانتزی از شهری روی ابرها بساز.",
        ],
        show_configuration_controls=False,
    )

    mo.vstack([
        mo.md("""
# ✦ PRIME AI
### یک چت هوشمند برای گفت‌وگو و ساخت تصویر

همه‌چیز از همین کادر چت انجام می‌شود:
- **Qwen3-32B (BF16)** برای گفت‌وگو و تشخیص درخواست ساخت تصویر
- **Qwen-Image-2512 (BF16)** برای تولید خودکار تصویر

فقط درخواست ساخت تصویرت را در چت بنویس؛ Prime AI تشخیص می‌دهد و تصویر را همان‌جا در گفت‌وگو نمایش می‌دهد.
برای اجرای مدل‌ها GPU را در تنظیمات Molab فعال کن. بار اول، وزن‌های عمومی مدل‌ها بدون API Key یا توکن Hugging Face دانلود می‌شوند.
"""),
        chat,
        mo.md("---\nPrime AI · یک سلول، یک چت، دو مدل محلی."),
    ])


if __name__ == "__main__":
    app.run()
