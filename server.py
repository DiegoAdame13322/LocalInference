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

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Automatic Disk Cleanup (safety for Codespaces)
# =====================================================
def check_disk_space(min_gb=2):
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"⚠️ Low disk space ({free_gb:.2f} GB). Clearing Hugging Face cache...")
        os.system("rm -rf ~/.cache/huggingface/*")

def background_health_monitor():
    while True:
        check_disk_space()
        time.sleep(600)  # every 10 minutes

threading.Thread(target=background_health_monitor, daemon=True).start()

# =====================================================
# Load Chat Model (Lightweight Qwen)
# =====================================================
print("Loading lightweight chat model (Qwen 1.5 0.5B Chat)…")
chat_model_name = "Qwen/Qwen1.5-0.1B-Chat"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
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
# Request Models
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
# Chat Endpoint (Fixed for Qwen 1.5 Chat)
# =====================================================
@app.post("/api/chat")
def chat_generate(req: ChatRequest):
    try:
        # Proper message template for Qwen 1.5 Chat
        prompt = (
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{req.message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Tokenize and run inference
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

        # Decode only newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        reply = chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Fallback in case of empty output
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
# Health + Static Routes
# =====================================================
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "models": ["chat: Qwen-0.5B-Chat", "summarization: DistilBART-6-6"]}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "AI Chat & Summarization API running!"}

# =====================================================
# Run FastAPI Server
# =====================================================
if __name__ == "__main__":
    # Get port from environment variable (Render provides this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting FastAPI server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
