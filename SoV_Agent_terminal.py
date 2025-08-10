# ----------------------------
# Terminal-ready SOV agent script
# ----------------------------
import os
import re
import time
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from googleapiclient.discovery import build
import cohere
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from transformers import pipeline

# Create the sentiment analysis pipeline just once at the top
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation = True,
    max_length = 512
)


# Load .env (make sure .env contains COHERE_API_KEY, YOUTUBE_API_KEY, HF_API_KEY)
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not (COHERE_API_KEY and YOUTUBE_API_KEY and HF_API_KEY):
    raise ValueError("Set COHERE_API_KEY, YOUTUBE_API_KEY and HF_API_KEY in your .env file")

# Clients / config
co = cohere.Client(COHERE_API_KEY)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
HF_BASE = "https://api-inference.huggingface.co/models/"
#SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

MAX_KEYWORDS = 10
VIDEOS_PER_KEYWORD = 5
COMMENTS_PER_VIDEO = 30

# Hard-coded competitor list (Atomberg + rivals)
COMPETITORS = ["Atomberg", "Crompton", "Havells", "Orient Electric", "Usha", "Bajaj"]

# ---------- HF inference wrapper with retry ----------
@retry(wait=wait_exponential(multiplier=1, min=1, max=12), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def hf_inference(model_name, payload, timeout=30):
    url = HF_BASE + model_name
    resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)
    if resp.status_code == 429:
        # raise to trigger tenacity retry/backoff
        raise requests.exceptions.RequestException("429 rate limit from HF")
    resp.raise_for_status()
    return resp.json()

def sentiment_of_text(text):
    """Return ('positive'|'neutral'|'negative', score) using HF pipeline."""
    try:
        # pipeline automatically tokenizes and truncates if needed
        out = sentiment_pipeline(text[:5000])
    except Exception as e:
        print(f"[WARN] Sentiment analysis failed: {e}")
        return "neutral", 0.0
    
    if isinstance(out, list) and out:
        label = out[0].get("label", "")
        score = float(out[0].get("score", 0.0))
        mapping = {
            "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
            "NEGATIVE": "negative", "NEUTRAL": "neutral", "POSITIVE": "positive"
        }
        return mapping.get(label, label.lower()), score

    return "neutral", 0.0

# ---------- Cohere keyword expansion ----------
def generate_keywords_cohere(base_keyword, max_k=MAX_KEYWORDS):
    prompt = (
        f"Generate up to {max_k-1} related search keywords for: '{base_keyword}'. "
        "Avoid brand names. Return a Python list like ['kw1','kw2', ...]."
    )
    try:
        resp = co.generate(model="command-xlarge", prompt=prompt, max_tokens=120, temperature=0.5)
        raw = resp.generations[0].text.strip()
    except Exception as e:
        print("[WARN] Cohere keyword expansion failed:", e)
        return [base_keyword]

    # try to parse Python literal list safely
    try:
        keywords = eval(raw, {"__builtins__":{}})
        if isinstance(keywords, list):
            kws = [base_keyword] + [k.strip() for k in keywords[:max_k-1]]
            # dedupe preserving order
            seen = set()
            out = []
            for k in kws:
                if k.lower() not in seen:
                    out.append(k)
                    seen.add(k.lower())
            return out[:max_k]
    except Exception:
        # fallback: split by newlines/commas
        parts = re.split(r'[\n,]+', raw)
        parts = [p.strip(" []'\"") for p in parts if p.strip()]
        kws = [base_keyword] + parts
        seen = set(); out=[]
        for k in kws:
            if k.lower() not in seen:
                out.append(k); seen.add(k.lower())
        return out[:max_k]

# ---------- YouTube helpers ----------
def search_youtube_videos(query, max_results=5000):
    resp = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results).execute()
    ids = [item["id"]["videoId"] for item in resp.get("items", [])]
    return ids

def get_video_details(video_ids):
    results = []
    if not video_ids:
        return results
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        resp = youtube.videos().list(part="snippet,statistics", id=",".join(batch)).execute()
        for it in resp.get("items", []):
            snip = it.get("snippet", {})
            stats = it.get("statistics", {})
            results.append({
                "video_id": it["id"],
                "title": snip.get("title", ""),
                "description": snip.get("description", ""),
                "channel": snip.get("channelTitle", ""),
                "view_count": int(stats.get("viewCount", 0)),
            })
    return results

def get_video_comments(video_id, max_comments=COMMENTS_PER_VIDEO):
    comments = []
    try:
        resp = youtube.commentThreads().list(videoId=video_id, part="snippet", maxResults=min(100, max_comments), textFormat="plainText").execute()
        for it in resp.get("items", []):
            comments.append(it["snippet"]["topLevelComment"]["snippet"].get("textDisplay", ""))
    except Exception:
        # comments can be disabled for some videos
        pass
    return comments

# ---------- NEW: explicit competitor mention analysis ----------
def analyze_video_mentions(video, competitors, do_sentiment=True):
    """
    video: dict with title, description, comments list, view_count
    competitors: list of brand strings
    returns: dict per brand: {'mentions': n, 'positive_mentions': m, 'views_attributed': views_if_mentioned}
    """
    title = video.get("title","")
    desc = video.get("description","")
    comments = video.get("comments", [])
    views = video.get("view_count", 0)

    text_blob = f"{title} {desc}".lower()
    brand_data = {b: {"mentions":0, "positive_mentions":0, "views_attributed":0} for b in competitors}

    # Title/description mentions: count occurrences
    for b in competitors:
        # use word boundaries for safety
        pattern = r"\b" + re.escape(b.lower()) + r"\b"
        count_td = len(re.findall(pattern, text_blob, flags=re.IGNORECASE))
        if count_td > 0:
            brand_data[b]["mentions"] += count_td
            brand_data[b]["views_attributed"] += views

    # Per-comment: check mention, call sentiment (only if comment mentions the brand)
    for c in comments:
        c_text = c.strip()
        if len(c_text) < 3:
            continue
        lower_c = c_text.lower()
        for b in competitors:
            if re.search(r"\b" + re.escape(b.lower()) + r"\b", lower_c, flags=re.IGNORECASE):
                # increment mention
                brand_data[b]["mentions"] += 1
                # optional: run sentiment for this comment and increment positive_mentions if positive
                if do_sentiment:
                    label, score = sentiment_of_text(c_text)
                    # throttle HF calls lightly
                    time.sleep(0.25 + random.random()*0.15)
                    if label == "positive":
                        brand_data[b]["positive_mentions"] += 1
    return brand_data

# ---------- SOV calculation (uses analyze_video_mentions explicitly) ----------
def calculate_sov(keywords, competitors=COMPETITORS, videos_per_kw=VIDEOS_PER_KEYWORD):
    # initialize aggregated stats
    agg = {b: {"mentions":0, "positive_mentions":0, "views":0, "videos_mentioned_in":0} for b in competitors}

    for kw in keywords:
        print(f"[INFO] fetching videos for '{kw}'")
        ids = search_youtube_videos(kw, max_results=videos_per_kw)
        details = get_video_details(ids)
        # get comments for each video and attach to details
        for d in details:
            d["comments"] = get_video_comments(d["video_id"], max_comments=COMMENTS_PER_VIDEO)

            # analyze mentions for this video (this is where we explicitly check the competitor list)
            per_brand = analyze_video_mentions(d, competitors, do_sentiment=True)
            for b, stats in per_brand.items():
                if stats["mentions"] > 0:
                    agg[b]["mentions"] += stats["mentions"]
                    agg[b]["positive_mentions"] += stats["positive_mentions"]
                    # views_attributed added above for td mentions
                    agg[b]["views"] += stats["views_attributed"]
                    agg[b]["videos_mentioned_in"] += (1 if stats["views_attributed"]>0 or stats["mentions"]>0 else 0)

    # Build DataFrame
    total_mentions = sum(agg[b]["mentions"] for b in competitors) or 1
    total_pos = sum(agg[b]["positive_mentions"] for b in competitors) or 1
    rows = []
    for b in competitors:
        m = agg[b]["mentions"]
        pm = agg[b]["positive_mentions"]
        v = agg[b]["views"]
        sov = (m / total_mentions) * 100
        pos_sov = (pm / total_pos) * 100 if total_pos>0 else 0.0
        rows.append({
            "brand": b,
            "mentions": m,
            "positive_mentions": pm,
            "views_attributed": v,
            "SoV (%)": round(sov,2),
            "Positive SoV (%)": round(pos_sov,2)
        })
    sov_df = pd.DataFrame(rows).sort_values("SoV (%)", ascending=False).reset_index(drop=True)
    return sov_df

# ---------- Plot merged chart ----------
def plot_sov_merged(sov_df, top_n=12):
    df = sov_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(df["brand"], df["SoV (%)"], label="SoV (%)", alpha=0.7)
    ax.set_ylabel("Share of Voice (%)")
    ax.set_xticklabels(df["brand"], rotation=45, ha="right")
    ax2 = ax.twinx()
    ax2.plot(df["brand"], df["Positive SoV (%)"], color="red", marker="o", label="Positive SoV (%)")
    ax2.set_ylabel("Positive SoV (%)")
    ax.set_title("Combined Share of Voice (merged across keywords)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    return fig

# ---------- Run pipeline ----------
if __name__ == "__main__":
    base_kw = "smart fans"
    keywords = generate_keywords_cohere(base_kw, max_k=MAX_KEYWORDS)
    print("[INFO] Keywords used:", keywords)

    sov_df = calculate_sov(keywords, competitors=COMPETITORS, videos_per_kw=VIDEOS_PER_KEYWORD)

    print("=== Share of Voice Table ===")
    print(sov_df.to_string(index=False))

    # Export CSV
    csv_path = "sov_results.csv"
    sov_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved results to {csv_path}")

    # Plot and export PNG
    fig = plot_sov_merged(sov_df)
    png_path = "sov_chart.png"
    fig.savefig(png_path)
    print(f"[INFO] Saved chart to {png_path}")