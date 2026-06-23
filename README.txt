# Signal — YouTube Comment Sentiment Analyzer

A Flask web app that analyzes YouTube video comments and breaks down 
what the audience actually thinks — not just a percentage, but weighted 
sentiment, keyword extraction, and the most impactful comments surfaced 
to the top.

## What it does

- Pulls top 100 comments from any YouTube video
- Filters out spam, bots, short noise comments, and non-English text
- Scores sentiment using VADER, weighted by comment likes
- Uses a HuggingFace RoBERTa transformer to verify the top comments 
  shown on the dashboard (handles sarcasm better than VADER alone)
- Extracts keywords separately for positive and negative comments
- Displays a vibe score (0-100), sentiment breakdown, keyword clouds, 
  and most impactful comments

## Setup

### 1. Clone the repo and install dependencies

git clone https://github.com/Vyhsnav/-signal-yt-sentiment-analyzer.git
cd signal-yt-sentiment-analyzer
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

### 2. Get your API keys

YouTube Data API v3 (free):
- Go to console.cloud.google.com
- Create a project, enable YouTube Data API v3
- Create credentials → API key

HuggingFace token (free):
- Create an account at huggingface.co
- Settings → Access Tokens → create a Read token

### 3. Create a .env file in the project root

YOUTUBE_API_KEY=your_youtube_key_here
HF_API_TOKEN=your_huggingface_token_here

### 4. Run it

python app.py

Then open http://127.0.0.1:5000 in your browser, paste a YouTube 
video URL, and hit analyze.

## Stack

Python, Flask, VADER, HuggingFace Transformers, YouTube Data API v3, 
NLTK, langdetect, Chart.js

## Known limitations

- Sarcasm and joke comments still trip up the model sometimes
- Non-English comments are filtered out rather than translated
- Keyword extraction is frequency-based so generic words can sneak in