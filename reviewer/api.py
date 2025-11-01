# reviewer/api.py
# FastAPI endpoint for the web MVP. No diacritics in comments.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, math

from novareview.prompts import build_prompt
from novareview.llm import ask_ollama
from novareview.heuristics import analyze_py

app = FastAPI(title="NovaReview API", version="0.1.0")

# CORS relaxed for local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],  # includes OPTIONS
    allow_headers=["*"],
)

class ReviewIn(BaseModel):
    code: str
    lang: str = "text"
    path: str = "pasted"

@app.get("/health")
def health():
    return {"ok": True}

def estimate_tokens(chars: int, chars_per_token: int = 4):
    return max(1, math.ceil(chars / max(1, chars_per_token)))

@app.post("/review")
def review(inp: ReviewIn):
    # config
    try:
        cfg = json.load(open(".aicodereviewrc.json", "r"))
    except Exception:
        cfg = {"model": "llama3.1", "max_context_chars": 4000, "guidelines": []}

    # deterministic layer (python)
    fixes, sugs = [], []
    if inp.lang in ("py", "python", "py3"):
        fixes, sugs = analyze_py(inp.code)

    # LLM prompt
    code = inp.code[: int(cfg.get("max_context_chars", 4000))]
    prompt = build_prompt(cfg.get("guidelines", []), inp.path, inp.lang, code)

    # call Ollama (JSON mode)
    llm_text, meta_raw = ask_ollama(cfg.get("model", "llama3.1"), prompt, timeout=90)

    # parse JSON safely
    data = {"summary": "(unstructured)", "comments": []}
    try:
        data = json.loads(llm_text) if llm_text else data
        if not isinstance(data, dict):
            data = {"summary": "(unstructured)", "comments": []}
    except Exception:
        pass

    # merge findings
    comments = []
    for f in fixes:
        comments.append({
            "lineOffset": f.get("lineOffset", 0),
            "message": f.get("message", ""),
            "suggestion": f.get("suggestion")
        })
    for s_ in sugs:
        comments.append({
            "lineOffset": s_.get("lineOffset", 0),
            "message": s_.get("message", "")
        })
    for c in (data.get("comments") or []):
        if isinstance(c, dict):
            comments.append(c)

    # simple metrics (so UI can show tokens/cost)
    prompt_chars = len(prompt)
    resp_chars   = len(llm_text or "")
    # prefer eval counts from Ollama if present
    p_tok = meta_raw.get("prompt_eval_count") or estimate_tokens(prompt_chars)
    r_tok = meta_raw.get("eval_count") or estimate_tokens(resp_chars)

    # price table (override in .aicodereviewrc.json if you want)
    prices = cfg.get("prices", {}).get(cfg.get("model", "llama3.1"), {})
    per_mtok = prices.get("per_mtok_usd", 0.0)
    cost_usd = round((p_tok + r_tok) / 1_000_000 * per_mtok, 6) if per_mtok else 0.0

    meta = {
        "model": cfg.get("model", "llama3.1"),
        "prompt_chars": prompt_chars,
        "response_chars": resp_chars,
        "prompt_tokens_est": p_tok,
        "response_tokens_est": r_tok,
        "chars_per_token": 4,
        "duration_ms": (meta_raw.get("total_duration") or 0) / 1_000_000,
        "price_per_mtok_usd": per_mtok,
        "cost_usd_est": cost_usd
    }

    return {
        "summary": data.get("summary", ""),
        "comments": comments,
        "meta": meta
    }
