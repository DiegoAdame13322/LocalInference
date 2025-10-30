# AI Chat & Summarization Web App 🤖

A beautiful web-based AI application featuring **Chat Generation** and **Text Summarization** powered by Hugging Face models.

## Features ✨

- 💬 **Chat Generation**: Interactive AI chat using Qwen/Qwen1.5-0.1B-Chat
- 📝 **Text Summarization**: Summarize long texts using DistilBART model
- 🎨 **Beautiful UI**: Modern gradient design with smooth animations
- 🌐 **Accessible**: Publicly deployable and accessible to everyone
- ⚡ **Fast**: Lightweight models optimized for quick responses

## Models Used

- **Chat**: `Qwen/Qwen1.5-0.5B-Chat` - Lightweight conversational AI
- **Summarization**: `sshleifer/distilbart-cnn-6-6` - Efficient text summarization

## Local Development

### Prerequisites
- Python 3.12+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DiegoAdame13322/LocalInference.git
cd LocalInference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python server.py
```

4. Open your browser to `http://localhost:8000`

## Deploy to Render 🚀

### Option 1: One-Click Deploy (Recommended)

1. Fork this repository to your GitHub account
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file
6. Click "Create Web Service"

### Option 2: Manual Deploy

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your repository
4. Configure:
   - **Name**: `ai-chat-summarization`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Instance Type**: Free or Starter (Starter recommended for better performance)

5. Click "Create Web Service"

### Important Notes for Render Deployment

- ⚠️ **First startup takes 5-10 minutes** as models download (1.5GB+)
- 💾 **Disk space**: Free tier has 512MB, models need ~1.5GB. Use **Starter plan** or higher
- 🔄 **Auto-sleep**: Free tier sleeps after 15min of inactivity, takes ~30s to wake up
- 🎯 **Recommendation**: Use **Starter plan ($7/month)** for:
  - More disk space
  - Better performance
  - No auto-sleep

## API Endpoints

### Chat Generation
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "What is machine learning?",
  "max_new_tokens": 150,
  "temperature": 0.7
}
```

### Text Summarization
```bash
POST /api/summarize
Content-Type: application/json

{
  "text": "Your long text here...",
  "max_length": 130,
  "min_length": 30
}
```

### Health Check
```bash
GET /api/health
```

## Project Structure

```
LocalInference/
├── server.py              # FastAPI backend with model loading
├── static/
│   └── index.html        # Frontend web interface
├── requirements.txt      # Python dependencies
├── render.yaml          # Render deployment config
├── runtime.txt          # Python version specification
└── README.md           # This file
```

## Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Models**: Hugging Face Transformers
- **Deployment**: Render

## Troubleshooting

### Models not loading on Render
- Upgrade to Starter plan for more disk space
- Check logs in Render dashboard

### Slow first response
- Models load on first request, subsequent requests are faster
- Consider keeping the service warm with periodic requests

### Out of memory errors
- Reduce `max_new_tokens` in chat requests
- Use Starter plan or higher for more RAM

## License

MIT License - feel free to use and modify!

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

Made with ❤️ using Hugging Face Transformers

