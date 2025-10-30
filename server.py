from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, uvicorn, os, subprocess, threading, shutil, time

# =====================================================
# FastAPI App Setup
# =====================================================
app = FastAPI(title="AI Chat + Summarization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Auto Disk Cleanup (for Codespaces)
# =====================================================
def check_disk_space(min_gb=2):
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"⚠️ Low disk space ({free_gb:.2f} GB). Clearing HuggingFace cache...")
        os.system("rm -rf ~/.cache/huggingface/*")

def background_health_monitor():
    while True:
        check_disk_space()
        time.sleep(600)

threading.Thread(target=background_health_monitor, daemon=True).start()

# =====================================================
# Load Chat Model (Qwen 1.5-0.5B-Chat)
# =====================================================
print("Loading Qwen 1.5-0.5B-Chat...")
chat_model_name = "Qwen/Qwen1.5-0.5B-Chat"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

# =====================================================
# Load Summarization Model
# =====================================================
print("Loading summarization model...")
summary_pipe = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device=0 if torch.cuda.is_available() else -1
)

# =====================================================
# API Schemas
# =====================================================
class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 80
    temperature: float = 0.7

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 100
    min_length: int = 25

# =====================================================
# Chat Endpoint
# =====================================================
@app.post("/api/chat")
def chat_generate(req: ChatRequest):
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{req.message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
        outputs = chat_model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=0.9,
            eos_token_id=chat_tokenizer.eos_token_id,
            pad_token_id=chat_tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        reply = chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not reply:
            reply = chat_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return {"success": True, "response": reply}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =====================================================
# Summarization Endpoint
# =====================================================
@app.post("/api/summarize")
def summarize_text(req: SummaryRequest):
    try:
        result = summary_pipe(
            req.text,
            max_length=req.max_length,
            min_length=min(req.min_length, req.max_length // 2),
            truncation=True,
        )
        key = "summary_text" if "summary_text" in result[0] else "generated_text"
        return {"success": True, "summary": result[0][key].strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =====================================================
# Health + Static
# =====================================================
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "models": ["Qwen-1.5-0.5B-Chat", "DistilBART-6-6"]}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "AI Chat & Summarization API running!"}

# =====================================================
# Run API + Cloudflare Tunnel
# =====================================================
if __name__ == "__main__":

    def run_api():
        print("🚀 Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    threading.Thread(target=run_api, daemon=True).start()

    # Start Cloudflare tunnel
    time.sleep(3)
    print("🌐 Starting Cloudflare Tunnel…")
    subprocess.run(["cloudflared", "tunnel", "--url", "http://localhost:8000"])
