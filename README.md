# Prime AI

Prime AI has a **single-cell Molab notebook** for chatting with a locally loaded Qwen3 model and generating images with FLUX.1 Schnell. It does not require a Hugging Face API token or another model-provider API key.

## Start Prime AI in Molab

Open the notebook file:

- [Open Prime AI in Molab](https://molab.marimo.io/github/fem-dev/prime-ai-gateway/blob/main/prime_ai_molab.py)
- [View the notebook source on GitHub](https://github.com/fem-dev/prime-ai-gateway/blob/main/prime_ai_molab.py)

In Molab:

1. Open the notebook and run its single cell.
2. For image generation, open the notebook resource settings and attach the available NVIDIA GPU.
3. The first chat or image request downloads the corresponding public model weights. The first FLUX download can be large and take time.
4. Chat with Qwen3 or enter an image description and press the image-generation button.

### Models

| Feature | Model | Where it runs |
| --- | --- | --- |
| Chat | `Qwen/Qwen3-0.6B` | In the Molab notebook runtime |
| Image generation | `black-forest-labs/FLUX.1-schnell` | In the Molab notebook runtime; GPU required |

No provider API key is configured. Public model weights are downloaded anonymously from the model host; that is a model-file download, not a call to a hosted inference API.

### Important Molab limitations

- Molab is an interactive notebook environment, not permanent hosting.
- Sessions can stop, and local runtime state or downloaded files may not persist.
- FLUX.1 Schnell is a large model. Enable a GPU before trying image generation.
- This notebook is intended for interactive use inside Molab. Do not expose the Molab runtime as a public API, proxy, or third-party compute service; follow the [Molab usage restrictions](https://molab.marimo.io/pages/molab/restrictions).

## Legacy API gateway

The repository also retains the earlier FastAPI API-gateway prototype in `main.py`. That prototype uses a provider token and is separate from the no-provider-key notebook above. It is not needed to run Prime AI in Molab.

