# Deploying this model as a Hugging Face Space (Gradio)

This repository contains a finetuned GPT-2 checkpoint `best_model.pt` and a small Gradio app (`app.py`) which will load the checkpoint (if present) and expose a simple text-generation UI.

Files added for the Space:
- `app.py` - Gradio app that loads `best_model.pt` (supports common state-dict wrappers) and exposes generation controls.
- `requirements.txt` - Python deps for a Hugging Face Space.

How to run locally

1. Create a virtual environment and install requirements.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app locally

```powershell
python app.py
```

3. Open http://127.0.0.1:7860 in your browser.

Deploy as a Hugging Face Space

1. Create a new Space on https://huggingface.co/spaces. Choose the Gradio template.
2. Push this repository (or copy files) into the Space. Make sure `best_model.pt` is included.
3. The Space will automatically install packages from `requirements.txt` and run `app.py`.

Notes and assumptions

- `app.py` attempts to load `best_model.pt` as either a state_dict or a full model object. If the checkpoint uses a different saving pattern, you might need to convert it to a Hugging Face compatible state dict or `model.save_pretrained()` format.
- For large models or GPU inference, configure the Space or environment accordingly (enable GPU for faster response).

If you want, I can also:

- Add a small wrapper to detect & use GPU when available.
- Add an example `space` metadata file (like `README.md` in the Space root) or create a `Dockerfile` for custom runtime.
