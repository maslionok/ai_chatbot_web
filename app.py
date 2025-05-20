# Import necessary libraries
import asyncio
import glob
import os
import shelve
import sys
import time
from urllib.parse import urljoin, urlparse

# Third-party libraries
import faiss
import httpx
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from openai import OpenAI
from PyPDF2 import PdfReader

# ─── Load configuration ───────────────────────────────────────────────────────
# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
MAGENTO_BASE_URL     = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE   = os.getenv("MAGENTO_STORE_CODE", "")
MAGENTO_BEARER_TOKEN = os.getenv("MAGENTO_BEARER_TOKEN", "")
DEFAULT_MAX_PAGES    = int(os.getenv("DEFAULT_MAX_PAGES", "200"))

# ─── Cache setup ──────────────────────────────────────────────────────────────
# Setup cache directory and database paths
CACHE_DIR  = os.getenv("CACHE_DIR", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB   = os.getenv("CACHE_DB", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_DB)
CACHE_DB_CHATWOOT   = os.getenv("CACHE_DB_CHATWOOT", "chatwoot")
CACHE_PATH_CHATWOOT = os.path.join(CACHE_DIR, CACHE_DB_CHATWOOT)

# ─── Initialize clients ───────────────────────────────────────────────────────
# Initialize OpenAI client and FAISS index
client = OpenAI(api_key=OPENAI_API_KEY)
faiss_index = None

# ─── Chatwoot configuration ───────────────────────────────────────────────────
# Load Chatwoot credentials and validate them
CHATWOOT_BASE_URL   = os.getenv("CHATWOOT_BASE_URL", "https://app.chatwoot.com")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_INBOX_ID   = os.getenv("CHATWOOT_INBOX_ID")
CHATWOOT_API_TOKEN  = os.getenv("CHATWOOT_API_TOKEN")
if not all([CHATWOOT_ACCOUNT_ID, CHATWOOT_INBOX_ID, CHATWOOT_API_TOKEN]):
    raise RuntimeError("Missing Chatwoot configuration: ensure ACCOUNT_ID, INBOX_ID, and API_TOKEN are set")

CHATWOOT_HEADERS = {
    "Content-Type": "application/json",
    "api_access_token": CHATWOOT_API_TOKEN
}

# ─── Track escalated conversations ───────────────────────────────────────────
# Dictionary to track conversations escalated to human agents
escalated_conversations = {}

# ─── Chatwoot helper functions ────────────────────────────────────────────────
def create_chatwoot_conversation(user_name: str = "Chatbot User"):
    """
    Create a new Chatwoot conversation and return its ID.
    - Generates a random email for the user.
    - Creates a contact in Chatwoot.
    - Starts a new conversation for the contact.
    """
    random_number = np.random.randint(100000, 999999)
    payload = {
        "name": user_name,
        "email": f"testclient{random_number}@example.com",
        "inbox_id": CHATWOOT_INBOX_ID
    }
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts"
    resp = requests.post(url, headers=CHATWOOT_HEADERS, json=payload)
    contact_data = resp.json()

    contact_id = contact_data["payload"]["contact"]["id"]
    source_id = contact_data["payload"]["contact"]["contact_inboxes"][0]["source_id"]

    print("✅ Contact created")

    # === 2. Create a conversation ===
    conversation_payload = {
        "source_id": source_id,
        "inbox_id": CHATWOOT_INBOX_ID,
        "contact_id": contact_id
    }
    r = requests.post(
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations",
        headers=CHATWOOT_HEADERS,
        json=conversation_payload
    )
    conversation_id = r.json()["id"]
    

    
    return conversation_id


def send_to_human_agent(message: str, user_name: str = "Chatbot User"):
    """
    Escalate a conversation to a human agent.
    - Reuses an existing conversation if available.
    - Filters out bot messages containing only "...".
    - Sends the user's message to the Chatwoot conversation.
    """
    if user_name in escalated_conversations:
        conv_id = escalated_conversations[user_name]
    else:
        conv_id = create_chatwoot_conversation(user_name)
        if not conv_id:
            print("❌ Skipping message send due to missing conversation ID.")
            return False
        escalated_conversations[user_name] = conv_id

    # Filter out bot messages containing only "..."
    filtered_message = "\n".join(
        line for line in message.split("\n") if not (line.startswith("bot:") and line.strip() == "bot: ...")
    )

    msg_payload = {"content": filtered_message, "message_type": "incoming"}
    msg_url = (
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/"
        f"{CHATWOOT_ACCOUNT_ID}/conversations/{conv_id}/messages"
    )
    resp = httpx.post(msg_url, headers=CHATWOOT_HEADERS, json=msg_payload)
    if resp.status_code not in (200, 201):
        print(f"❌ Chatwoot message send failed: {resp.status_code} {resp.text}")
        return False
    return True

# ─── Magento headers ───────────────────────────────────────────────────────────
# Headers for Magento API requests
MAGENTO_HEADERS = {
    "Authorization": f"Bearer {MAGENTO_BEARER_TOKEN}",
    "Content-Type": "application/json"
}

# ─── Crawling utilities ──────────────────────────────────────────────────────
async def fetch_url(session, url):
    """
    Fetch a URL asynchronously.
    - Extracts and cleans HTML content.
    - Returns the URL, text content, and parsed BeautifulSoup object.
    """
    try:
        r = await session.get(url, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            soup = BeautifulSoup(r.text, 'html.parser')
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            return url, soup.get_text(separator=' ', strip=True), soup
    except Exception:
        pass
    return url, None, None

async def crawl_website_async(start_url, max_pages=DEFAULT_MAX_PAGES):
    """
    Crawl a website asynchronously.
    - Visits up to `max_pages` pages starting from `start_url`.
    - Extracts text content from each page.
    """
    visited, to_visit, texts = set(), [start_url], []
    base = "{0.scheme}://{0.netloc}".format(urlparse(start_url))
    async with httpx.AsyncClient(follow_redirects=True) as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            fetched, text, soup = await fetch_url(session, url)
            if text:
                texts.append(text)
                visited.add(fetched)
                for link in soup.find_all("a", href=True):
                    href = urljoin(base, link['href']).split('#')[0]
                    if base in href and href not in visited and href not in to_visit:
                        to_visit.append(href)
    return "\n\n".join(texts)

# ─── Magento API fetcher ─────────────────────────────────────────────────────
async def fetch_magento_products(page_size=100):
    """
    Fetch products from Magento API.
    - Paginates through the API to retrieve all products.
    - Returns a list of product dictionaries.
    """
    items = []
    async with httpx.AsyncClient() as session:
        page = 1
        while True:
            url = f"{MAGENTO_BASE_URL}/rest/{MAGENTO_STORE_CODE}/V1/products"
            params = {"searchCriteria[currentPage]": page, "searchCriteria[pageSize]": page_size}
            resp = await session.get(url, headers=MAGENTO_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json().get("items", [])
            if not batch:
                break
            items.extend(batch)
            page += 1
    return items

# ─── Data flattening ─────────────────────────────────────────────────────────
def product_to_text(product: dict) -> str:
    """
    Convert a Magento product dictionary into a human-readable text format.
    - Includes SKU, name, price, status, and custom attributes.
    """
    out = [
        f"SKU: {product.get('sku')}",
        f"Name: {product.get('name')}",
        f"Price: {product.get('price')}",
        f"Status: {'Enabled' if product.get('status') == 1 else 'Disabled'}"
    ]
    for attr in product.get("custom_attributes", []):
        out.append(f"{attr.get('attribute_code')}: {attr.get('value')}")
    return "\n".join(out)

# ─── PDF loader ──────────────────────────────────────────────────────────────
def load_pdfs(folder="extra_info"):
    """
    Load and extract text from all PDF files in the specified folder.
    - Returns a list of document texts.
    """
    docs = []
    if not os.path.isdir(folder): return docs
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                docs.append("\n".join(pages))
        except Exception as e:
            print(f"PDF load error {path}: {e}")
    return docs

# ─── Text splitting & embedding ───────────────────────────────────────────────
def split_text(text, max_tokens=300):
    """
    Split a large text into smaller chunks of up to `max_tokens` words.
    """
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks, batch_size=100):
    """
    Generate embeddings for text chunks using OpenAI API.
    - Processes chunks in batches for efficiency.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(model="text-embedding-3-small", input=chunks[i:i+batch_size])
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings).astype("float32")

# ─── Language detection & ChatGPT call ────────────────────────────────────────
def detect_language(question):
    """
    Detect the language of a given question using OpenAI's ChatGPT.
    - Returns the detected language name.
    """
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"Detect language and reply with only the language name."},
                  {"role":"user","content":question}]
    )
    return resp.choices[0].message.content.strip()

def ask_chatgpt(question, chunks, model="gpt-4.1-mini"):
    """
    Query ChatGPT with a question and context.
    - Combines context chunks into a single prompt.
    - Returns the model's response.
    """
    context = "\n\n".join(chunks)
    lang = detect_language(question)
    sys_msg = (f"You are a precise assistant. Answer ONLY based on provided context. Respond in {lang}. "
               "Short, step-by-step if needed, only rely on context.")
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

# ─── Flask app setup ─────────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def home():
    """
    Render the home page.
    """
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat API requests.
    - Processes user questions and retrieves answers from ChatGPT.
    - Supports crawling, Magento data, and human-agent escalation.
    """
    data      = request.json or {}
    question  = data.get('question', '').strip()
    model     = data.get('model', 'gpt-4.1-mini')
    crawl_url = data.get('url')
    max_pages = int(data.get('max_pages', DEFAULT_MAX_PAGES))
    user_name = data.get('user_name', 'Chatbot User')
    history   = data.get('history', [])  # Retrieve conversation history

    # Check if conversation is already escalated
    if user_name in escalated_conversations:
        full_conversation = "\n".join(
            [f"{msg['who']}: {msg['text']}" for msg in history if not (msg['who'] == 'bot' and msg['text'] == "...")]
        )
        # send only last message to human agent and remove "user:" prefix
        last_message = full_conversation.split("\n")[-1]
        if last_message.startswith("user: "):
            last_message = last_message[6:]

        send_to_human_agent(last_message, user_name)
        return jsonify({
            "answer": "Your message has been forwarded to a human agent. They will assist you shortly."
        }), 200

    # Human-Agent handoff detection
    triggers = [
        # English
        "human",
        "agent",
        "support",
        "operator",
        "representative",
        "live person",
        "real person",
        "customer service",
        "help",
        "assistance",
        "talk to a real person",
        "speak to an agent",
        "connect me to someone",
        "chat with support",
        "I need help from a real person",
        "I want to talk to someone",

        # German
        "Mensch",
        "Agent",
        "Support",
        "Operator",
        "Vertreter",
        "Live-Mitarbeiter",
        "echter Mensch",
        "Kundenservice",
        "Hilfe",
        "mit jemandem sprechen",
        "zum menschlichen Support",
        "an einen Mitarbeiter weiterleiten",
        "Ich möchte mit einem Menschen sprechen",
        "Bitte verbinden Sie mich mit einem Mitarbeiter",

        # Polish
        "człowiek",
        "agent",
        "wsparcie",
        "operator",
        "przedstawiciel",
        "obsługa klienta",
        "pomoc",
        "rozmowa z człowiekiem",
        "połącz mnie z żywą osobą",
        "przełącz na konsultanta",
        "chcę z kimś porozmawiać",
        "porozmawiaj z człowiekiem",
        "Chcę rozmawiać z prawdziwą osobą",
        "Proszę o przekierowanie do konsultanta",
    ]
    if any(t in question.lower() for t in triggers):
        full_conversation = "\n".join(
            [f"{msg['who']}: {msg['text']}" for msg in history if not (msg['who'] == 'bot' and msg['text'] == "...")]
        )
        send_to_human_agent(full_conversation, user_name)
        return jsonify({
            "answer": "Your message has been forwarded to a human agent. They will assist you shortly."
        }), 200

    with shelve.open(CACHE_PATH) as db:
        # Magento
        mag_key = f"magento||{MAGENTO_STORE_CODE}"
        if mag_key in db:
            mag_chunks, mag_embs = db[mag_key]
        else:
            prods = asyncio.run(fetch_magento_products(page_size=200))
            texts = [product_to_text(p) for p in prods] + load_pdfs()
            mag_chunks = []
            for t in texts:
                mag_chunks.extend(split_text(t))
            mag_embs = embed_chunks(mag_chunks)
            db[mag_key] = (mag_chunks, mag_embs)

        combined_chunks = list(mag_chunks)
        combined_embs = mag_embs

        # Crawl
        if crawl_url:
            crawl_key = f"crawl||{crawl_url}||{max_pages}"
            if crawl_key in db:
                crawl_chunks, crawl_embs = db[crawl_key]
            else:
                crawled = asyncio.run(crawl_website_async(crawl_url, max_pages))
                crawl_chunks = split_text(crawled)
                crawl_embs = embed_chunks(crawl_chunks)
                db[crawl_key] = (crawl_chunks, crawl_embs)
            combined_chunks.extend(crawl_chunks)
            combined_embs = np.vstack([combined_embs, crawl_embs])

    # Build FAISS index
    global faiss_index
    if faiss_index is None or faiss_index.ntotal != combined_embs.shape[0]:
        idx = faiss.IndexFlatL2(combined_embs.shape[1])
        idx.add(combined_embs)
        faiss_index = idx

    # Query
    resp_emb = client.embeddings.create(model="text-embedding-3-small", input=[question])
    q_emb   = np.array([resp_emb.data[0].embedding]).astype("float32")
    dist, inds = faiss_index.search(q_emb, 10)
    top = [combined_chunks[i] for i in inds[0]]

    answer = ask_chatgpt(question, top, model)
    return jsonify({"answer": answer})


@app.route('/chatwoot/webhook', methods=['POST'])
def receive_chatwoot_reply():
    """
    Handle incoming Chatwoot webhook events.
    - Processes outgoing messages and maps them to chat sessions.
    """
    data = request.json
    message_type = data.get("message_type")
    content = data.get("content")
    sender_type = data.get("sender", {}).get("type")
    chatwoot_conversation_id = str(data.get("conversation", {}).get("id", "default"))

    print(f"📨 Webhook: type={message_type}, sender={sender_type}, content={content}")

    if content and message_type == "outgoing":  # Accept all outgoing messages
        print(f"📨 Outgoing message: {content}")
        with shelve.open(CACHE_PATH_CHATWOOT) as db:
            # Find the associated chat session
            mapped_session = None
            for key in db.keys():
                if key.startswith("chatwoot_map||") and db[key] == chatwoot_conversation_id:
                    mapped_session = db.get(f"session_map||{chatwoot_conversation_id}")
                    break
            
            if mapped_session:
                # Save to mapped session
                history_key = f"chat_history||{mapped_session}"
                messages = db.get(history_key, [])
                messages.append({"role": "bot", "content": content})
                db[history_key] = messages
                print(f"✅ Added to mapped session {mapped_session}")
            else:
                # Store in general chat history if no mapping exists
                chat_key = "global_chat_history"
                messages = db.get(chat_key, [])
                messages.append({"role": "bot", "content": content})
                db[chat_key] = messages
                print("✅ Added to global chat history")
    return jsonify({"status": "received"}), 200

@app.route('/api/messages/<conversation_id>', methods=['GET'])
def get_messages(conversation_id):
    """
    Retrieve chat messages for a specific conversation.
    - Merges session-specific and global messages.
    """
    with shelve.open(CACHE_PATH_CHATWOOT) as db:
        # Get messages specific to this session
        history_key = f"chat_history||{conversation_id}"
        messages = db.get(history_key, [])
        
        
        # Also get any global messages
        global_messages = db.get("global_chat_history", [])
        if global_messages:
            # Merge messages without duplicates
            seen = set(msg["content"] for msg in messages)
            for msg in global_messages:
                if msg["content"] not in seen:
                    messages.append(msg)
                    seen.add(msg["content"])
            # Save merged messages to session
            db[history_key] = messages
            # Clear global messages
            db["global_chat_history"] = []
            
    return jsonify(messages)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)





