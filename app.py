import os
import re
from flask import Flask, render_template, request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from analyzer import analyze

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

app = Flask(__name__)
youtube = build('youtube','v3',developerKey=API_KEY)

#helpers
def extract_video_id(url):
    """ Pull video ID from url format """

    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',   
        r'youtu\.be\/([0-9A-Za-z_-]{11})', 
    ]

    for pattern in patterns:
        match = re.search(pattern,url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_id,max_results=100):
    """ Pull top-level comments """

    comments = []
    request_obj = youtube.commentThreads().list(
        part = 'snippet',
        videoId = video_id,
        maxResults = min(max_results,100),
        order = 'relevance',
        textFormat = 'plainText'
    )

    response = request_obj.execute()

    for item in response.get('items',[]):
        snippet = item['snippet']['topLevelComment']['snippet']

        comments.append({
            'text': snippet['textDisplay'],
            'likes': snippet['likeCount'],
        })

    return comments

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    url = request.form.get('video_url','').strip()
    video_id = extract_video_id(url)

    if not video_id:
        return render_template('index.html', error="Coudn'nt find a valid Youtube video.")
    
    try:
        video_response = youtube.videos().list(part='snippet',id=video_id).execute()

        if not video_response['items']:
            return render_template('index.html', error="Video not found.")
        
        video_title = video_response['items'][0]['snippet']['title']
        raw_comments = fetch_comments(video_id)
        if not raw_comments:
            return render_template('index.html', error="No comment found in video.")
        
        result = analyze(raw_comments)
        result['video_title'] = video_title

        return render_template('dashboard.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=f"Something went wrong: {str(e)}")


if __name__== '__main__':
    app.run(debug=True)
