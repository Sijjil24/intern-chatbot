import json
import os
import re
import time
import uuid

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from kb import KnowledgeBase

load_dotenv()

# Folder that contains this file. Using it makes file paths work on Vercel too.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
kb = KnowledgeBase(os.path.join(BASE_DIR, "knowledge_base", "faqs.json"))


def _find_key():
    for name in ("GROK_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        v = os.getenv(name, "").strip().strip('"').strip("'")
        if v:
            return v
    return ""


GROK_API_KEY = _find_key()
# Auto-detect the provider from the key prefix
if GROK_API_KEY.startswith("gsk_"):
    PROVIDER = "Groq"
    GROK_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROK_MODEL = os.getenv("GROK_MODEL", "openai/gpt-oss-20b")
else:
    PROVIDER = "xAI Grok"
    GROK_URL = "https://api.x.ai/v1/chat/completions"
    GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")

print(f"AI mode: {PROVIDER} ({GROK_MODEL})" if GROK_API_KEY else "AI mode: OFFLINE - no API key found in .env")

SYSTEM_PROMPT = """You are Internee Assistant, a friendly, smart AI assistant for interns of Internee.pk (Pakistan's virtual internship platform).

You can chat naturally like ChatGPT: greet people, answer follow-up questions, explain concepts, and help with the technical side of internship tasks (Python, Flask, web development, APIs, Firebase, AI, debugging, resumes, interview prep).

For questions about Internee.pk itself (tasks, submissions, certificates, tracks, mentorship, contact), use ONLY the OFFICIAL INFO below. If the answer is not there, say you don't have that detail and suggest checking the task's Guidelines tab or contacting Internee.pk at +92 312 3023645. Never invent deadlines, stipends, dates, policies or links.

Style: warm, clear and concise. Use short paragraphs, bullet points and code blocks when helpful. Match the user's language (English, Urdu or Roman Urdu). Never ask for passwords or API keys.

OFFICIAL INFO:
"""


def system_message():
    facts = "\n\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in kb.entries)
    return SYSTEM_PROMPT + facts


# ---------- Optional Firebase logging ----------
db = None
try:
    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")  # for Vercel / servers
    cred_path = os.path.join(BASE_DIR, os.getenv("FIREBASE_CREDENTIALS", "serviceAccountKey.json"))
    if cred_json or os.path.exists(cred_path):
        import firebase_admin
        from firebase_admin import credentials, firestore

        cert = credentials.Certificate(json.loads(cred_json) if cred_json else cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cert)
        db = firestore.client()
        print("Firebase connected.")
    else:
        print("Firebase skipped (no credentials found).")
except Exception as exc:
    print("Firebase disabled:", exc)


def log(collection, data):
    if not db:
        return
    try:
        data["created_at"] = time.time()
        db.collection(collection).add(data)
    except Exception as exc:
        print("Firebase log failed:", exc)


def build_messages(question, history):
    msgs = [{"role": "system", "content": system_message()}]
    for m in history[-12:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"])[:3000]})
    msgs.append({"role": "user", "content": question})
    return msgs


def stream_grok(messages):
    """Yield text chunks from Grok (OpenAI-compatible streaming)."""
    r = requests.post(
        GROK_URL,
        headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROK_MODEL, "messages": messages, "temperature": 0.5, "stream": True},
        stream=True,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Grok API error {r.status_code}: {r.text[:300]}")
    # Fix: the API does not send a charset, so requests guesses Latin-1 and garbles
    # apostrophes, emojis and Urdu text. Force UTF-8.
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, ValueError):
            continue
        if delta:
            yield delta


GREETING = re.compile(r"^\s*(hi|hello|hey|salam|assalam.*|aoa|good (morning|afternoon|evening))\W*$", re.I)


def offline_answer(question):
    """Used only when no API key is configured."""
    if GREETING.match(question):
        return "Hello! I'm the Internee.pk assistant. Ask me about tasks, submissions, certificates or mentorship."
    hits = kb.search(question, k=1)
    if hits:
        return hits[0]["answer"]
    return (
        "I couldn't find that. Check the Guidelines tab on your task, or contact Internee.pk at +92 312 3023645.\n\n"
        "(Add your GROK_API_KEY to .env to unlock full AI answers.)"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({"ai": bool(GROK_API_KEY), "provider": PROVIDER if GROK_API_KEY else None})


@app.route("/api/topics")
def topics():
    return jsonify([e["question"] for e in kb.entries])


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    question = (body.get("message") or "").strip()[:1500]
    history = body.get("history") or []
    session_id = body.get("session_id") or str(uuid.uuid4())
    if not question:
        return jsonify({"error": "Please type a question."}), 400

    @stream_with_context
    def generate():
        parts = []
        try:
            if GROK_API_KEY:
                for chunk in stream_grok(build_messages(question, history)):
                    parts.append(chunk)
                    yield chunk
            else:
                text = offline_answer(question)
                parts.append(text)
                yield text
        except Exception as exc:
            print("Chat error:", exc)
            msg = "Sorry, I couldn't get an answer right now. Please try again in a moment."
            parts.append(msg)
            yield msg
        log("chat_logs", {"session_id": session_id, "question": question, "answer": "".join(parts)})

    return Response(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/feedback", methods=["POST"])
def feedback():
    body = request.get_json(silent=True) or {}
    log("feedback", {"session_id": body.get("session_id"), "question": body.get("question"), "helpful": bool(body.get("helpful"))})
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)