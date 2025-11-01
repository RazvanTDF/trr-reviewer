# novareview/llm.py
# Simple Ollama client with JSON mode and sane timeouts.

import requests

def ask_ollama(model: str, prompt: str, timeout: int = 60):
    url = "http://localhost:11434/api/generate"
    # JSON mode = returneaza DOAR JSON (fara cod fences), mult mai usor de parsat
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",       # <- important!
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()  # {'response': '...json...', 'eval_count': ...}
    text = data.get("response", "") or ""
    # meta utile (tokens ~ eval_count, etc.)
    meta = {
        "model": model,
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration": data.get("total_duration"),  # ns
    }
    return text, meta
