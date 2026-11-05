import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Prime AI — local-model chat and image generation for Molab.
    # Public model weights are downloaded anonymously; no HF/API token is used.
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path

    import marimo as mo

    INSTALL_MARKER = Path.home() / ".prime_ai_deps_installed"
    if not INSTALL_MARKER.exists():
        packages = [
            "transformers>=4.51.0",
            "diffusers>=0.33.0",
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

    # Keep loaded pipelines between reactive reruns of this notebook cell.
    state = globals().setdefault(
        "_prime_ai_state",
        {"qwen": None, "qwen_tokenizer": None, "flux": None,
         "lock": threading.RLock(), "last_image": None},
    )

    QWEN_ID = "Qwen/Qwen3-0.6B"
    FLUX_ID = "black-forest-labs/FLUX.1-schnell"

    def load_qwen():
        with state["lock"]:
            if state["qwen"] is None:
                tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
                kwargs = {"device_map": "auto"}
                if torch.cuda.is_available():
                    kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model = AutoModelForCausalLM.from_pretrained(QWEN_ID, **kwargs)
                state["qwen_tokenizer"] = tokenizer
                state["qwen"] = model
            return state["qwen"], state["qwen_tokenizer"]

    def answer(messages, config):
        model, tokenizer = load_qwen()
        chat_messages = []
        for item in messages:
            role = getattr(item, "role", "user")
            text = getattr(item, "content", "")
            if isinstance(text, str) and role in ("system", "user", "assistant"):
                chat_messages.append({"role": role, "content": text})
        if not chat_messages:
            return "لطفاً یک پیام متنی بفرست."
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with state["lock"], torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    chat = mo.ui.chat(
        answer,
        prompts=["سلام! خودت را معرفی کن.", "یک ایده برای پروژه برنامه‌نویسی بده."],
        show_configuration_controls=False,
    )

    image_prompt = mo.ui.text_area(
        label="توضیح تصویر",
        placeholder="مثلاً: یک منظره فانتزی از شهر آینده در غروب، سبک نقاشی دیجیتال",
        rows=3,
    )
    make_image = mo.ui.button(label="ساخت تصویر با FLUX.1 Schnell", kind="primary")
    image_hint = mo.md("برای تولید تصویر، GPU را از تنظیمات منابع Molab فعال کن. بار اول دانلود مدل ممکن است زمان‌بر باشد.")

    mo.vstack([
        mo.md("""
# ✦ PRIME AI
### چت و تولید تصویر با اجرای محلی مدل‌ها

رابط Prime AI با سلول‌های هماهنگ Marimo اجرا می‌شود. برای چت از **Qwen3-0.6B** و برای تصویر از
**FLUX.1 Schnell** استفاده می‌کند. مدل‌ها در همین نشست اجرا می‌شوند؛ درخواست‌ها
به API سرویس‌های واسطه ارسال نمی‌شوند.

**نیازی به API Key یا توکن Hugging Face نیست.** وزن‌های عمومی مدل‌ها هنگام
اولین استفاده به‌صورت ناشناس دانلود می‌شوند. برای تولید تصویر، GPU را از
تنظیمات منابع Molab فعال کن. نشست‌های Molab موقت‌اند و ممکن است پس از پایان
نشست، مدل‌های دانلودشده و تاریخچه از بین بروند.
"""),
        mo.md("## گفت‌وگو با Qwen3"),
        chat,
        mo.md("## تولید تصویر با FLUX.1 Schnell"),
        image_prompt,
        make_image,
        image_hint,
        mo.md("---\nPrime AI · مدل‌ها روی محیط همین Notebook اجرا می‌شوند."),
    ])
    return mo, torch, state, FLUX_ID, image_prompt, make_image


@app.cell
def _(mo, torch, state, FLUX_ID, image_prompt, make_image):
    # UI values must be read in a different cell from the one that creates them.
    mo.stop(not make_image.value)
    prompt_text = image_prompt.value.strip()
    if not prompt_text:
        image_status = mo.md("ابتدا توضیح تصویر را بنویس.")
    elif not torch.cuda.is_available():
        image_status = mo.md("برای FLUX.1 Schnell به GPU نیاز است. GPU را در تنظیمات Molab فعال کن و دوباره اجرا کن.")
    else:
        try:
            from diffusers import FluxPipeline
            with state["lock"]:
                if state["flux"] is None:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    pipe = FluxPipeline.from_pretrained(FLUX_ID, torch_dtype=dtype)
                    pipe.enable_model_cpu_offload()
                    state["flux"] = pipe
                result = state["flux"](
                    prompt=prompt_text,
                    height=1024,
                    width=1024,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                )
                state["last_image"] = result.images[0]
            image_status = mo.vstack([
                mo.md("### تصویر آماده شد"),
                mo.image(state["last_image"], width=512),
            ])
        except Exception as exc:
            image_status = mo.vstack([
                mo.md("### تولید تصویر ناموفق بود"),
                mo.code(str(exc)[-3000:]),
            ])
    image_status


if __name__ == "__main__":
    app.run()
