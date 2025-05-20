const btn       = document.getElementById('chat-button');
const container = document.getElementById('chat-container');
const closeBtn  = document.getElementById('close-btn');
const box       = document.getElementById('chat-box');
const input     = document.getElementById('chat-input');
const send      = document.getElementById('send-btn');

// Toggle chat visibility
btn.onclick = () => {
  container.classList.toggle('hidden');
  if (!container.classList.contains('hidden')) {
    setTimeout(() => input.focus(), 150);
  }
};
closeBtn.onclick = () => {
  container.classList.add('hidden');
  // Optional: clear chat on close for a fresh start
  setTimeout(() => { box.innerHTML = ''; }, 200);
};

// Store conversation history
const conversationHistory = [];

// Append a message; returns the div
function append(who, text) {
  const div = document.createElement('div');
  div.className   = 'message ' + who;
  div.textContent = text;
  div.style.opacity = 0;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  setTimeout(() => { div.style.transition = 'opacity 0.25s'; div.style.opacity = 1; }, 10);
  conversationHistory.push({ who, text }); // Save to history
  return div;
}

// Send a message to the backend AI
async function postMessage(text) {
  append('user', text);
  input.value = '';

  // show loading dots
  const loader = append('bot', '...');

  // Use default website and page limit
  const payload = {
    url: window.location.origin,
    max_pages: 200,
    model: 'gpt-4.1-mini',
    question: text,
    history: conversationHistory // Include conversation history
  };

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const resJSON = await res.json();
    if (resJSON.crawl_done) {
      btn.style.background = "linear-gradient(135deg, green 60%, darkgreen 100%)";
    }
    loader.textContent = resJSON.answer;
    conversationHistory.push({ who: 'bot', text: resJSON.answer }); // Save bot's response
  } catch (e) {
    loader.textContent = 'Error: ' + e.message;
  }
}

// WebSocket setup for Chatwoot
const chatwootSocket = new WebSocket('wss://app.chatwoot.com/cable');

chatwootSocket.onopen = () => {
  console.log('Connected to Chatwoot WebSocket');
  // Replace 'conversation_id' with the actual conversation ID from the backend
  const conversationId = window.chatwootConversationId || null;
  if (conversationId) {
    chatwootSocket.send(JSON.stringify({
      command: 'subscribe',
      identifier: JSON.stringify({
        channel: 'RoomChannel',
        id: conversationId
      })
    }));
  }
};

chatwootSocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'ping' || !data.message) return;

  const messageData = data.message;
  if (!messageData || !messageData.data) return;

  const message = messageData.data;

  // print(message.content)
  
  // Check if it's a message from an agent (message_type: 1 indicates outgoing message from agent)
  if (message.message_type === 1 && message.content) {
    // Get the sender name, fallback to "Human Agent" if not available
    const senderName = message.sender?.name || "Human Agent";
    // Display the message in the chat with the agent's name
    append('agent', `${senderName}: ${message.content}`);
    // Auto-scroll to the latest message
    box.scrollTop = box.scrollHeight;
  }
};

chatwootSocket.onclose = () => {
  console.log('Disconnected from Chatwoot WebSocket');
};

const sessionId = 'chat_' + Date.now();

let lastMessageCount = 0;
let isFirstLoad = true;

// Fetch all messages from Flask
async function fetchMessages() {
  try {
    const res = await fetch(`/api/messages/${sessionId}`);
    const messages = await res.json();
    // add console.log to see the messages
    console.log("Fetched messages:", messages);

    
    if (!messages || messages.length === 0) {
      return;
    }

    // Only remove non-persistent loading indicators
    const loaders = box.querySelectorAll('.message.bot:not(.persist)');
    loaders.forEach(loader => {
      if (loader.textContent === '...') {
        loader.remove();
      }
    });

    // Reset messages if count decreased (new session)
    if (messages.length < lastMessageCount) {
      box.innerHTML = '';
      lastMessageCount = 0;
    }

    // Add new messages
    if (messages.length > lastMessageCount) {
      const newMessages = messages.slice(lastMessageCount);
      newMessages.forEach(msg => {
        append(msg.role === 'user' ? 'user' : 'bot', msg.content);
      });
      lastMessageCount = messages.length;
    }
  } catch (err) {
    console.error("Polling error:", err);
  }
}

// Hook up send
send.onclick = () => {
  if (input.value.trim()) postMessage(input.value);
};
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send.click();
  }
});

// Poll for new messages every 200ms
setInterval(fetchMessages, 200);
fetchMessages();
