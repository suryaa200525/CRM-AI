from flask import Flask, render_template, request
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import mimetypes
import os
import random
import re
import time
import traceback
import uuid
 
load_dotenv()
 
app = Flask(__name__)
 
# Gemini inline audio requests are limited to roughly 20 MB in total
app.config["MAX_CONTENT_LENGTH"] = 18 * 1024 * 1024
 
# ------------------------------------------------
# Gemini setup
# ------------------------------------------------
 
api_key = os.getenv("GEMINI_API_KEY")
 
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file")
 
client = genai.Client(api_key=api_key)
 
# Models are tried in this order. To try a different primary model without
# editing the code, add GEMINI_MODEL=<model name> to your .env file.
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = ["gemini-3.1-flash-lite"]
MODELS = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
 
# Status codes that mean "try again" (rate limit / server busy)
RETRYABLE_CODES = (429, 500, 503, 504)
ATTEMPTS_PER_MODEL = 3
 
UPLOAD_FOLDER = "uploads"
 
FIELDS = [
    "Sentiment",
    "Customer Interest",
    "Customer Concern",
    "Follow-up Required",
    "Follow-up Reason",
    "Customer Intent",
    "Summary",
]
 
PROMPT = """
Analyze this customer conversation for a CRM system.
 
The conversation may be in English, Tamil, or Tamil-English mixed language.
 
Understand the conversation accurately.
 
DO NOT provide the transcript.
 
Return ONLY these 7 fields:
 
Sentiment
Customer Interest
Customer Concern
Follow-up Required
Follow-up Reason
Customer Intent
Summary
 
Rules:
 
Sentiment must be exactly one of:
 
Positive
Negative
Neutral
 
Follow-up Required must be exactly:
 
Yes
No
 
Customer Intent must be exactly one of:
 
Interested
Not Interested
Needs More Information
 
If there is no specific customer concern, write:
 
No specific concern identified.
 
If follow-up is not required, write:
 
No follow-up required.
 
Keep the summary short.
 
Use EXACTLY this format:
 
Sentiment: <answer>
Customer Interest: <answer>
Customer Concern: <answer>
Follow-up Required: <answer>
Follow-up Reason: <answer>
Customer Intent: <answer>
Summary: <answer>
"""
 
 
# ------------------------------------------------
# Helpers
# ------------------------------------------------
 
def analyse_audio(audio_data, mime_type):
    """Call Gemini with retries, then fall back to the next model if busy."""
 
    last_error = None
 
    for model in MODELS:
 
        for attempt in range(ATTEMPTS_PER_MODEL):
 
            try:
 
                print(f"Calling {model} (attempt {attempt + 1})...")
 
                return client.models.generate_content(
                    model=model,
                    contents=[
                        PROMPT,
                        types.Part.from_bytes(
                            data=audio_data,
                            mime_type=mime_type
                        )
                    ],
                    config=types.GenerateContentConfig(temperature=0.2)
                )
 
            except errors.APIError as e:
 
                last_error = e
 
                # Model name does not exist -> go to the next model
                if e.code == 404:
                    print(f"{model} was not found. Trying next model.")
                    break
 
                # Busy / rate limited -> wait and retry
                if e.code in RETRYABLE_CODES:
 
                    if attempt < ATTEMPTS_PER_MODEL - 1:
 
                        delay = 3 * (2 ** attempt) + random.uniform(0, 2)
 
                        print(
                            f"{model} busy (HTTP {e.code}). "
                            f"Retrying in {delay:.0f} seconds..."
                        )
 
                        time.sleep(delay)
 
                    continue
 
                # Anything else (bad request, bad key, etc.) will not fix itself
                raise
 
        print(f"Switching away from {model}.")
 
    raise last_error
 
 
def parse_analysis(text):
    """Turn Gemini's 'Label: value' reply into a dictionary."""
 
    # Gemini sometimes wraps labels in markdown bold (**Sentiment:**)
    text = text.replace("*", "")
 
    labels = "|".join(re.escape(field) for field in FIELDS)
    result = {}
 
    for field in FIELDS:
 
        pattern = (
            r"^\s*" + re.escape(field) + r"\s*:\s*(.*?)"
            r"(?=^\s*(?:" + labels + r")\s*:|\Z)"
        )
 
        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
 
        value = match.group(1).strip() if match else ""
 
        result[field] = value or "Not available"
 
    return result
 
 
def normalise_choice(value, allowed):
    """Fix case and trailing full stops, e.g. 'positive.' -> 'Positive'."""
 
    cleaned = value.strip().rstrip(".").strip()
 
    for option in allowed:
        if cleaned.lower() == option.lower():
            return option
 
    return value
 
 
# ------------------------------------------------
# Routes
# ------------------------------------------------
 
@app.route("/")
def home():
    return render_template("index.html")
 
 
@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="The audio file is too large. Please upload a file under 18 MB."
    ), 413
 
 
@app.route("/analyse", methods=["POST"])
def analyse():
 
    customer_name = request.form.get("customer_name", "").strip()
    audio = request.files.get("audio")
 
    if not customer_name:
        return render_template(
            "index.html",
            error="Please enter the customer name."
        ), 400
 
    if audio is None or audio.filename == "":
        return render_template(
            "index.html",
            customer_name=customer_name,
            error="Please select an audio file."
        ), 400
 
    # Save audio with a unique, safe file name
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
    safe_name = secure_filename(audio.filename) or "audio"
    audio_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4().hex}_{safe_name}"
    )
    audio.save(audio_path)
 
    print("Customer Name:", customer_name)
    print("Audio File:", audio.filename)
    print("Saved As:", audio_path)
 
    try:
 
        # Read audio file
        with open(audio_path, "rb") as file:
            audio_data = file.read()
 
        # Get MIME type (browsers sometimes send a generic type)
        mime_type = audio.mimetype
 
        if not mime_type or mime_type == "application/octet-stream":
            mime_type = mimetypes.guess_type(safe_name)[0] or "audio/mpeg"
 
        print("MIME Type:", mime_type)
 
        # Call Gemini (with retry + fallback)
        response = analyse_audio(audio_data, mime_type)
 
        analysis = (response.text or "").strip()
 
        if not analysis:
            raise ValueError("Gemini returned an empty response.")
 
        print("Gemini analysis successful.")
        print("\nGemini Response:")
        print(analysis)
 
        # Extract CRM fields
        data = parse_analysis(analysis)
 
        sentiment = normalise_choice(
            data["Sentiment"],
            ["Positive", "Negative", "Neutral"]
        )
        interest = data["Customer Interest"]
        concern = data["Customer Concern"]
        follow_up = normalise_choice(
            data["Follow-up Required"],
            ["Yes", "No"]
        )
        follow_up_reason = data["Follow-up Reason"]
        intent = normalise_choice(
            data["Customer Intent"],
            ["Interested", "Not Interested", "Needs More Information"]
        )
        summary = data["Summary"]
 
        print("\nFinal CRM Data:")
        print("Sentiment:", sentiment)
        print("Interest:", interest)
        print("Concern:", concern)
        print("Follow-up:", follow_up)
        print("Follow-up Reason:", follow_up_reason)
        print("Intent:", intent)
        print("Summary:", summary)
 
        return render_template(
            "index.html",
            customer_name=customer_name,
            sentiment=sentiment,
            interest=interest,
            concern=concern,
            follow_up=follow_up,
            follow_up_reason=follow_up_reason,
            intent=intent,
            summary=summary
        )
 
    except errors.APIError as e:
 
        print("\n========== GEMINI API ERROR ==========")
        print(type(e).__name__, e.code)
        print(str(e))
        print("======================================")
 
        if e.code in RETRYABLE_CODES:
            message = (
                "Gemini is busy right now. "
                "Please wait a few minutes and try again."
            )
        else:
            message = (
                f"Gemini returned an error (code {e.code}). "
                "Please check the Flask terminal for details."
            )
 
        return render_template(
            "index.html",
            customer_name=customer_name,
            error=message
        ), 503
 
    except Exception as e:
 
        print("\n========== ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        traceback.print_exc()
        print("============================")
 
        return render_template(
            "index.html",
            customer_name=customer_name,
            error=(
                "Audio analysis failed. "
                "Please check the Flask terminal for the error."
            )
        ), 500
 
 
if __name__ == "__main__":
    # debug=True is for local development only; turn it off in production
    app.run(debug=True)
 