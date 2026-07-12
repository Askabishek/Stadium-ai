# ⚽ StadiumAI — FIFA World Cup 2026

## AI-Powered Stadium Operations & Fan Experience Platform

An intelligent, multimodal GenAI solution that enhances stadium operations and the overall tournament experience for fans, organizers, volunteers, and venue staff during the FIFA World Cup 2026.

---

## 🎯 Problem Statement

Managing 80,000+ fans across multiple stadiums during the FIFA World Cup presents massive challenges: navigation confusion, crowd bottlenecks, language barriers for international visitors, slow emergency response, and limited real-time operational intelligence. Traditional systems fail at scale.

## 💡 Solution

**StadiumAI** is a single multimodal AI pipeline that provides:

- **AI Navigation** — Fans find seats, facilities, and exits with natural language directions
- **Real-Time Crowd Monitoring** — Zone-level density tracking with AI-predicted hotspots
- **Multilingual Assistance** — Instant translation in 14+ languages with audio output
- **Voice Interface** — Speak questions in any language, get instant AI answers
- **Semantic Search** — RAG-powered knowledge base for stadium info
- **Operational Intelligence** — AI insights, emergency response plans, and staff recommendations
- **Accessibility First** — Wheelchair routes, accessible facilities, screen reader support

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Main LLM (Text + Vision) | Groq Llama 4 Scout 17B |
| Speech-to-Text | Groq Whisper Large V3 |
| Text-to-Speech | gTTS |
| RAG / Semantic Search | Chroma + Sentence-Transformers (all-MiniLM-L6-v2) |
| Database | SQLite |
| Frontend | Streamlit |
| Maps | Folium |
| Charts | Plotly |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
stadiumai/
├── app.py                    # Streamlit UI (7 pages)
├── pipeline.py               # Core multimodal pipeline
├── tools/
│   ├── navigation_tool.py    # AI wayfinding & facility finder
│   ├── crowd_tool.py         # Crowd density & predictions
│   ├── multilingual_tool.py  # Translation & multilingual chat
│   ├── voice_tool.py         # Whisper STT + gTTS TTS
│   ├── sql_tool.py           # NL-to-SQL queries
│   └── rag_tool.py           # Semantic search with Chroma
├── data/
│   └── seed_data.py          # Synthetic data generator (8 stadiums)
├── tests/
│   └── test_tools.py         # Unit tests (pytest)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Deployment

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/StadiumAI.git
cd StadiumAI

# Install dependencies
pip install -r requirements.txt

# Set environment variable
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file: `app.py`
4. In Advanced Settings → Secrets:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy!

---

## ✅ Features

### For Fans
- 🧭 **Find Facility** — Locate restrooms, food, first aid, ATMs, exits
- 🗺️ **Get Directions** — AI step-by-step navigation with accessibility options
- 🌐 **Multilingual Chat** — Ask questions in 14+ languages
- 🎤 **Voice Queries** — Speak and get instant answers
- 🔍 **Smart Search** — Semantic search over stadium knowledge

### For Organizers & Staff
- 👥 **Crowd Monitor** — Real-time zone density with treemap visualization
- 🔮 **AI Predictions** — 2-hour crowd flow forecasts
- 🚨 **Emergency Response** — AI-generated response plans with multilingual alerts
- 📊 **Operational Insights** — Actionable recommendations for staff deployment
- 💬 **Feedback Analysis** — Sentiment breakdown and category trends
- 🗄️ **NL Database Query** — Ask questions in plain English

### Accessibility
- ♿ Wheelchair-accessible route prioritization
- 🔊 Audio output for all responses
- ⌨️ Full keyboard navigation
- 📱 Responsive design
- 🏷️ ARIA labels and skip navigation

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tools --cov-report=term-missing
```

---

## 🔒 Security

- All API keys stored as environment variables (never hardcoded)
- SQL injection prevention with query validation
- Dangerous SQL operations (DROP, DELETE, UPDATE, INSERT) are blocked
- Input sanitization on all user inputs
- No sensitive data in version control

---

## 📊 Data

The app auto-generates synthetic data on first run:
- 8 FIFA World Cup 2026 stadiums (USA, Mexico, Canada)
- 500+ facility locations per stadium
- Real-time crowd density simulation
- Match schedules, transport options, and fan feedback
- Multilingual announcements

---

## 🌍 Supported Languages

English, Spanish, French, Arabic, Hindi, Portuguese, German, Japanese, Korean, Mandarin Chinese, Italian, Dutch, Russian, Turkish

---

## 📄 License

MIT License

---

*Built for the Virtual PromptWars Hackathon — Build with AI 2026*
