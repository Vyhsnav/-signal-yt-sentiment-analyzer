import re
import os
import math
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from collections import Counter
from huggingface_hub import InferenceClient
import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)
from nltk.corpus import stopwords

load_dotenv()

STOP_WORDS = set(stopwords.words('english'))
hf_client = InferenceClient(token=os.getenv("HF_API_TOKEN"))


def clean_text(text):
    """Strip URLs, mentions, special chars, lowercase."""
    text = re.sub(r'http\S+', '', text)         # remove URLs
    text = re.sub(r'@\w+', '', text)            # remove mentions
    text = re.sub(r'[^\w\s]', '', text)         # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # collapse whitespace
    return text.lower()

#checking if comment is valid
def is_valid_comment(text):
    # Check if the comment is empty or None
    if not text or not text.strip():
        return False
    
    #check if commnt is too short
    if len(text.split())<4:
        return False
    
    #check if comment is emoji/symbols
    if re.match(r'^[\W_]+$', text):
        return False
    
    #check if comment is in english
    try:
        if detect(text) != 'en':
            return False
    except LangDetectException:
        return False

    return True

#text preprocessing
def preprocess(comments):
    """
    Input: list of dicts - {text,likes}
    Output: list of dicts - {text,cleaned,likes}
    """

    seen=set()
    filtered=[]
    for c in comments:
        raw=c['text']
        cleaned=clean_text(raw)
        if not is_valid_comment(cleaned):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        filtered.append({
            'text': raw,
            'cleaned': cleaned,
            'likes': c['likes']
        })
    return filtered

#weighted sentiment score

analyzer = SentimentIntensityAnalyzer()

def score_comment(comment):
    """add VADER compund score and weighted score to a dict"""
    compound = analyzer.polarity_scores(comment['cleaned'])['compound']
    weight = math.log1p(comment['likes'])   
    return {**comment, 'compound': compound, 'weighted': compound * weight}

def classify(compound):
    """classify based on VADER compund score"""
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'

def compute_vibe_score(comments):
    """
    weighted average sentiment mapped to 0-100 scale
    pure pos = 100, pure neg = 0
    """

    total_weight = sum(math.log1p(c['likes']) + 1 for c in comments)
    weighted_sum = sum(c['compound'] * (math.log1p(c['likes']) + 1) for c in comments)

    avg = weighted_sum / total_weight if total_weight > 0 else 0
    return round((avg + 1) / 2 * 100, 2)  # map from [-1,1] to [0,100]

#keyword extraction
def extract_keywords(comments,sentiment, top_n=10):
    """Extract top meaningful keywords for a given sentiment class"""

    words=[]
    for c in comments:
        if classify(c['compound'])==sentiment:
            tokens=c['cleaned'].split()
            words+=[w for w in tokens if w not in STOP_WORDS and len(w)>3]
    freq=Counter(words)
    return freq.most_common(top_n)

#top comments
def get_top_comments_both(comments, n=3, pool_size=15):
    """
    Pull the most-liked comments overall, classify each with the HF model,
    then split into top n positive and top n negative by likes.
    """
    candidates = sorted(comments, key=lambda c: c['likes'], reverse=True)[:pool_size]

    pos, neg = [], []
    for c in candidates:
        hf_label = hf_classify(c['text'])
        if hf_label == 'positive':
            pos.append(c)
        elif hf_label == 'negative':
            neg.append(c)

    return pos[:n], neg[:n]

def hf_classify(text):
    """
    Sends a single comment to a RoBERTa sentiment model via Hugging Face's
    Inference Providers (current routing as of 2026).
    Returns 'positive', 'negative', or 'neutral'. Returns None on failure.
    """
    try:
        result = hf_client.text_classification(
            text,
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        # result is a list of label/score dicts, sorted by score descending
        top = result[0]
        return top.label.lower()
    except Exception as e:
        print(f"HF API error: {e}")
        return None


#main pipeline
def analyze(raw_comments):
    """Full pipeline: preprocess, score, classify, compute vibe, extract keywords
    Input: list of dicts - {text,likes}
    Output: dict with vibe score, keyword lists, and comment breakdown
    """

    clean=preprocess(raw_comments)
    total=len(raw_comments)
    kept=len(clean)
    dropped=total-kept

    #2-score each comment
    scored=[score_comment(c) for c in clean]

    #3-bucket counts
    counts=Counter(classify(c['compound']) for c in scored)

    #4-vibe score
    vibe=compute_vibe_score(scored)

    #5-keywords
    pos_keywords=extract_keywords(scored,'positive')
    neg_keywords=extract_keywords(scored,'negative')
    
    #6-top comments
    top_pos, top_neg = get_top_comments_both(scored)
    

    return {
        'vibe_score': vibe,
        'counts': dict(counts),
        'total': total,
        'kept': kept,
        'dropped': dropped,
        'pos_keywords': pos_keywords,
        'neg_keywords': neg_keywords,
        'top_positive': top_pos,
        'top_negative': top_neg
    }
