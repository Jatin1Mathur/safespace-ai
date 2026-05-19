"""
SafeSpace AI - Desktop Professional UI
"""

import gradio as gr
import requests
from datetime import datetime

BACKEND_URL = "http://localhost:8000/ask"


def call_backend(message: str) -> tuple[str, str, str, str]:
    try:
        res = requests.post(BACKEND_URL, json={"message": message}, timeout=60)
        res.raise_for_status()
        data = res.json()
        response = data.get("response") or "I'm here to listen. Could you tell me more?"
        tool = data.get("tool_called", "None")
        mood = data.get("mood", "neutral")
        risk = data.get("risk", "NONE")
        return response, tool, mood, risk
    except requests.exceptions.ConnectionError:
        return "⚠️ Backend is offline. Please run: `cd backend && python main.py`", "Error", "neutral", "NONE"
    except Exception as e:
        return f"Something went wrong: {str(e)}", "Error", "neutral", "NONE"


MOOD_EMOJI = {
    "anxious": "😟", "depressed": "😔", "angry": "😠",
    "lonely": "💙", "stressed": "😰", "happy": "😊",
    "neutral": "😐"
}

RISK_COLOR = {
    "HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#2563eb", "NONE": "#16a34a"
}


def chat(user_message: str, history: list, tool_log: str, mood_display: str, session_stats: str):
    if not user_message.strip():
        return history, "", tool_log, mood_display, session_stats

    response, tool_called, mood, risk = call_backend(user_message)
    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})

    timestamp = datetime.now().strftime("%H:%M:%S")
    tool_log = f"[{timestamp}] Tool: {tool_called} | Mood: {mood} | Risk: {risk}\n" + tool_log

    emoji = MOOD_EMOJI.get(mood, "😐")
    mood_display = f"{emoji} {mood.capitalize()}"

    # Update session stats
    msg_count = len([m for m in history if m["role"] == "user"])
    session_stats = f"Messages: {msg_count} · Risk: {risk} · Tool: {tool_called}"

    return history, "", tool_log, mood_display, session_stats


def clear_chat():
    return [], "", "", "😐 Neutral", "Messages: 0"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@200;300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg: #0e1117;
    --bg-panel: #141920;
    --bg-card: #1a2230;
    --bg-input: #1e2738;
    --border: #2a3448;
    --border-light: #1e2a3a;
    --text: #e8eef8;
    --text-sub: #8494b0;
    --text-muted: #4a5878;
    --green: #10b981;
    --green-dim: #064e35;
    --green-glow: rgba(16,185,129,0.15);
    --blue: #3b82f6;
    --blue-dim: #1e3a6e;
    --red: #ef4444;
    --red-dim: #4a1010;
    --amber: #f59e0b;
    --amber-dim: #4a3010;
    --accent: #10b981;
    --radius: 12px;
    --radius-lg: 18px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 48px rgba(0,0,0,0.6);
    --glow: 0 0 40px rgba(16,185,129,0.08);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 24px 40px !important;
}

/* ── Top nav bar ── */
.ss-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 16px;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 24px;
}

.ss-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.ss-brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #064e35, #10b981);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(16,185,129,0.3);
}

.ss-brand-name {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.3px;
}

.ss-brand-name span {
    color: var(--green) !important;
}

.ss-brand-tag {
    font-size: 0.65rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 300 !important;
    margin-top: 2px;
}

.ss-nav-chips {
    display: flex;
    gap: 8px;
    align-items: center;
}

.ss-nav-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    border: 1px solid;
}

.nc-green { background: var(--green-glow); border-color: #064e35; color: var(--green); }
.nc-blue  { background: rgba(59,130,246,0.1); border-color: var(--blue-dim); color: var(--blue); }
.nc-red   { background: rgba(239,68,68,0.08); border-color: var(--red-dim); color: var(--red); }

.nc-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Desktop 2-column layout ── */
.ss-layout {
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 20px;
    align-items: start;
}

/* ── Left: main chat area ── */
.ss-main {}

/* ── Quick prompts ── */
.ss-quickbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}

.ss-qbtn {
    background: var(--bg-card) !important;
    color: var(--text-sub) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    padding: 5px 14px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    font-family: 'Outfit', sans-serif !important;
    white-space: nowrap;
}

.ss-qbtn:hover {
    background: var(--green-glow) !important;
    border-color: #064e35 !important;
    color: var(--green) !important;
    transform: translateY(-1px);
}

/* ── Chat window ── */
#chatbot {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow), var(--glow) !important;
}

/* ── Input panel ── */
.ss-input-wrap {
    margin-top: 14px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.ss-input-wrap:focus-within {
    border-color: var(--green);
    box-shadow: 0 0 0 3px var(--green-glow);
}

#msg-input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#msg-input textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    padding: 16px 18px 8px !important;
    resize: none !important;
    min-height: 90px !important;
}

#msg-input textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}

#msg-input textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

.ss-input-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px 12px;
    border-top: 1px solid var(--border-light);
}

.ss-input-hint {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-weight: 300;
    font-family: 'JetBrains Mono', monospace;
}

#send-btn {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 9px 28px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 12px rgba(16,185,129,0.35) !important;
}

#send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(16,185,129,0.5) !important;
}

#clear-btn {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 9px 18px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}

#clear-btn:hover {
    border-color: var(--text-muted) !important;
    color: var(--text-sub) !important;
}

/* ── Right sidebar ── */
.ss-sidebar {
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: sticky;
    top: 20px;
}

.ss-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
}

.ss-card-title {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}

/* Mood card */
.ss-mood-display {
    font-size: 1.8rem;
    text-align: center;
    padding: 8px 0 4px;
    letter-spacing: -1px;
}

#mood-label {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-align: center !important;
}

#mood-label textarea, #mood-label input {
    background: transparent !important;
    border: none !important;
    color: var(--green) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    padding: 4px 0 !important;
}

/* Stats card */
#session-stats {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#session-stats textarea, #session-stats input {
    background: transparent !important;
    border: none !important;
    color: var(--text-sub) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 0 !important;
    line-height: 1.8 !important;
}

/* Disclaimer card */
.ss-disclaimer-text {
    font-size: 0.72rem;
    color: var(--text-muted);
    line-height: 1.6;
    font-weight: 300;
}

.ss-disclaimer-text strong {
    color: var(--amber);
    font-weight: 500;
}

/* Tool log */
#tool-log textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    line-height: 1.7 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Footer */
.ss-footer {
    text-align: center;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border-light);
}

.ss-footer p {
    font-size: 0.65rem !important;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.08em;
}
"""

with gr.Blocks(title="SafeSpace AI") as demo:

    # ── Navbar ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="ss-navbar">
        <div class="ss-brand">
            <div class="ss-brand-icon">🌿</div>
            <div>
                <div class="ss-brand-name">Safe<span>Space</span> AI</div>
                <div class="ss-brand-tag">Mental Health · Medical Support</div>
            </div>
        </div>
        <div class="ss-nav-chips">
            <span class="ss-nav-chip nc-green"><span class="nc-dot"></span>MedGemma</span>
            <span class="ss-nav-chip nc-blue"><span class="nc-dot"></span>Llama 3.2</span>
            <span class="ss-nav-chip nc-red"><span class="nc-dot"></span>Emergency Ready</span>
        </div>
    </div>
    """)

    # ── 2-column layout ──────────────────────────────────────────────────────
    gr.HTML('<div class="ss-layout"><div class="ss-main">')

    # Quick prompts
    gr.HTML("""
    <div class="ss-quickbar">
        <button class="ss-qbtn" onclick="setMsg('I am feeling anxious today')">😟 Feeling anxious</button>
        <button class="ss-qbtn" onclick="setMsg('Find hospitals in Hof')">🏥 Find hospitals</button>
        <button class="ss-qbtn" onclick="setMsg('Find pharmacy near me')">💊 Find pharmacy</button>
        <button class="ss-qbtn" onclick="setMsg('I am having trouble sleeping')">😴 Sleep issues</button>
        <button class="ss-qbtn" onclick="setMsg('I feel overwhelmed')">🌊 Overwhelmed</button>
        <button class="ss-qbtn" onclick="setMsg('Give me a breathing exercise')">🫁 Breathing</button>
        <button class="ss-qbtn" onclick="setMsg('Show my mood history')">📊 Mood history</button>
        <button class="ss-qbtn" onclick="setMsg('Give me crisis hotline numbers')">📞 Hotlines</button>
    </div>
    <script>
    function setMsg(text) {
        const ta = document.querySelector('#msg-input textarea');
        if (!ta) return;
        const nv = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
        nv.set.call(ta, text);
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.focus();
    }
    </script>
    """)

    # Chat
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        height=500,
        layout="bubble",
        placeholder="<div style='text-align:center;padding:100px 24px;'><div style='font-family:Playfair Display,serif;font-size:1.5rem;font-weight:400;color:#2a3a50;font-style:italic;'>What's on your mind today?</div><div style='font-size:0.68rem;color:#2a3a50;letter-spacing:0.18em;text-transform:uppercase;margin-top:12px;font-family:JetBrains Mono,monospace;opacity:0.5;'>safe · confidential · always here</div></div>",
    )

    # Input panel
    with gr.Group(elem_classes="ss-input-wrap"):
        msg = gr.Textbox(
            placeholder="Share what's on your mind... I'm here to listen",
            show_label=False,
            elem_id="msg-input",
            lines=3,
            max_lines=8,
        )
        gr.HTML("""
        <div class="ss-input-bar">
            <span class="ss-input-hint">↵ enter to send · end-to-end private</span>
        </div>
        """)
        with gr.Row():
            send_btn = gr.Button("Send →", elem_id="send-btn", scale=4)
            clear_btn = gr.Button("New session", elem_id="clear-btn", scale=1)

    gr.HTML('</div><div class="ss-sidebar">')

    # Mood card
    gr.HTML('<div class="ss-card"><div class="ss-card-title">Current Mood</div>')
    mood_label = gr.Textbox(
        value="😐 Neutral",
        elem_id="mood-label",
        show_label=False,
        interactive=False,
        lines=1,
    )
    gr.HTML('</div>')

    # Session stats card
    gr.HTML('<div class="ss-card"><div class="ss-card-title">Session Stats</div>')
    session_stats = gr.Textbox(
        value="Messages: 0",
        elem_id="session-stats",
        show_label=False,
        interactive=False,
        lines=2,
    )
    gr.HTML('</div>')

    # Disclaimer card
    gr.HTML("""
    <div class="ss-card">
        <div class="ss-card-title">⚠ Important</div>
        <div class="ss-disclaimer-text">
            SafeSpace AI is <strong>not a substitute</strong> for professional care.<br><br>
            In emergencies call:<br>
            <strong style="color:#10b981">112</strong> Europe<br>
            <strong style="color:#10b981">911</strong> USA<br>
            <strong style="color:#10b981">999</strong> UK
        </div>
    </div>
    """)

    # Tool log card
    gr.HTML('<div class="ss-card"><div class="ss-card-title">Agent Log</div>')
    tool_log = gr.Textbox(
        elem_id="tool-log",
        show_label=False,
        lines=6,
        interactive=False,
        placeholder="Activity log...",
    )
    gr.HTML('</div>')

    gr.HTML('</div></div>')  # close sidebar + layout

    gr.HTML("""
    <div class="ss-footer">
        <p>SAFESPACE AI · MEDGEMMA + LLAMA 3.2 · ALL CONVERSATIONS PRIVATE & ENCRYPTED</p>
    </div>
    """)

    # ── State ────────────────────────────────────────────────────────────────
    history_state = gr.State([])
    tool_log_state = gr.State("")

    send_btn.click(
        fn=chat,
        inputs=[msg, history_state, tool_log_state, mood_label, session_stats],
        outputs=[chatbot, msg, tool_log, mood_label, session_stats],
    ).then(
        fn=lambda h, t: (h, t),
        inputs=[chatbot, tool_log],
        outputs=[history_state, tool_log_state],
    )

    msg.submit(
        fn=chat,
        inputs=[msg, history_state, tool_log_state, mood_label, session_stats],
        outputs=[chatbot, msg, tool_log, mood_label, session_stats],
    ).then(
        fn=lambda h, t: (h, t),
        inputs=[chatbot, tool_log],
        outputs=[history_state, tool_log_state],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg, tool_log, mood_label, session_stats],
    ).then(
        fn=lambda: ([], ""),
        outputs=[history_state, tool_log_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )