import os
from datetime import datetime

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/ask")

GENERAL_SERVICE_MESSAGE = (
    "The assistant service is starting or temporarily unavailable. "
    "Please try again in a moment."
)

TECHNICAL_ERROR_KEYWORDS = [
    "twilio",
    "traceback",
    "exception",
    "connectionerror",
    "httperror",
    "api error",
    "system start error",
    "backend error",
    "failed to start",
    "localhost",
    "python main.py",
    "internal server error",
    "500",
]


MOOD_EMOJI = {
    "anxious": "😟",
    "depressed": "😔",
    "angry": "😠",
    "lonely": "💙",
    "stressed": "😰",
    "happy": "😊",
    "neutral": "😐",
}

RISK_META = {
    "HIGH": ("High Risk", "risk-high", "Immediate support recommended"),
    "MEDIUM": ("Medium Risk", "risk-medium", "Monitor closely"),
    "LOW": ("Low Risk", "risk-low", "Supportive guidance"),
    "NONE": ("No Risk", "risk-none", "Normal support mode"),
}


def is_technical_error_message(text: str) -> bool:
    if not text:
        return False

    lower_text = str(text).lower()
    return any(keyword in lower_text for keyword in TECHNICAL_ERROR_KEYWORDS)


def call_backend(message: str) -> tuple[str, str, str, str]:
    try:
        res = requests.post(BACKEND_URL, json={"message": message}, timeout=300)
        res.raise_for_status()
        data = res.json()

        response = data.get("response") or "I'm here to listen. Could you tell me more?"
        tool = data.get("tool_called", "None")
        mood = data.get("mood", "neutral")
        risk = data.get("risk", "NONE")

        if is_technical_error_message(response):
            response = GENERAL_SERVICE_MESSAGE
            tool = "System"
            mood = "neutral"
            risk = "NONE"

        if is_technical_error_message(tool):
            tool = "System"

        return response, tool, mood, risk

    except requests.exceptions.ConnectionError:
        return GENERAL_SERVICE_MESSAGE, "System", "neutral", "NONE"

    except requests.exceptions.Timeout:
        return GENERAL_SERVICE_MESSAGE, "System", "neutral", "NONE"

    except requests.exceptions.RequestException:
        return GENERAL_SERVICE_MESSAGE, "System", "neutral", "NONE"

    except Exception:
        return GENERAL_SERVICE_MESSAGE, "System", "neutral", "NONE"


def make_risk_badge(risk: str) -> str:
    label, class_name, helper = RISK_META.get(risk, RISK_META["NONE"])

    return f"""
    <div class="risk-badge {class_name}">
        <div>
            <span class="risk-dot"></span>
            <strong>{label}</strong>
        </div>
        <small>{helper}</small>
    </div>
    """


def chat(
    user_message: str,
    history: list,
    tool_log: str,
    mood_display: str,
    session_stats: str,
    risk_status: str,
):
    if not user_message or not user_message.strip():
        return history, "", tool_log, mood_display, session_stats, risk_status

    response, tool_called, mood, risk = call_backend(user_message.strip())

    history = history or []

    history.append(
        {
            "role": "user",
            "content": user_message.strip(),
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    timestamp = datetime.now().strftime("%H:%M:%S")
    tool_log = (
        f"[{timestamp}] Agent: {tool_called} | Mood: {mood} | Risk: {risk}\n"
        + (tool_log or "")
    )

    emoji = MOOD_EMOJI.get(mood, "😐")
    mood_display = f"{emoji} {mood.capitalize()}"

    msg_count = len(history) // 2
    session_stats = f"Messages: {msg_count}\nRisk: {risk}\nAgent: {tool_called}"
    risk_status = make_risk_badge(risk)

    return history, "", tool_log, mood_display, session_stats, risk_status


def clear_chat():
    return (
        [],
        "",
        "",
        "😐 Neutral",
        "Messages: 0\nRisk: NONE\nAgent: None",
        make_risk_badge("NONE"),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --page: #07110f;
    --page-2: #0b141f;
    --panel: rgba(17, 24, 39, 0.78);
    --panel-strong: rgba(15, 23, 42, 0.95);
    --card: rgba(20, 31, 45, 0.72);
    --card-hover: rgba(25, 39, 56, 0.92);
    --input: rgba(8, 15, 27, 0.9);
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(16, 185, 129, 0.35);
    --text: #edf7f4;
    --text-soft: #b8c7d9;
    --muted: #77879f;
    --muted-2: #536178;
    --green: #22c55e;
    --green-2: #10b981;
    --mint: #7dd3fc;
    --blue: #60a5fa;
    --amber: #f59e0b;
    --red: #f43f5e;
    --radius: 22px;
    --radius-sm: 14px;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
    --glow: 0 0 80px rgba(16, 185, 129, 0.13);
}

* {
    box-sizing: border-box;
}

body,
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 32rem),
        radial-gradient(circle at 85% 15%, rgba(96, 165, 250, 0.16), transparent 28rem),
        linear-gradient(135deg, var(--page), var(--page-2)) !important;
    color: var(--text) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 26px 26px 40px !important;
}

footer,
.api-docs {
    display: none !important;
}

.ss-shell {
    position: relative;
}

.ss-shell::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.6), transparent 70%);
}

.ss-header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px;
    align-items: center;
    padding: 20px;
    border: 1px solid var(--border);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(16,36,35,0.72));
    box-shadow: var(--shadow), var(--glow);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 16px;
}

.brand-icon {
    width: 54px;
    height: 54px;
    display: grid;
    place-items: center;
    border-radius: 18px;
    background: linear-gradient(135deg, #064e3b, #10b981 62%, #7dd3fc);
    box-shadow: 0 18px 45px rgba(16, 185, 129, 0.32);
    font-size: 1.5rem;
}

.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.65rem, 3vw, 2.35rem);
    line-height: 1;
    letter-spacing: -0.04em;
    margin: 0;
}

.brand-title span {
    color: var(--green);
}

.brand-subtitle {
    margin-top: 8px;
    color: var(--text-soft);
    font-size: 0.86rem;
}

.status-strip {
    display: flex;
    gap: 9px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-soft);
    background: rgba(255, 255, 255, 0.045);
    font-size: 0.76rem;
    font-weight: 600;
}

.status-pill::before {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 14px currentColor;
}

.pill-green {
    color: var(--green);
}

.pill-blue {
    color: var(--blue);
}

.pill-red {
    color: var(--red);
}

.hero-card {
    margin-bottom: 18px;
    padding: 18px 20px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(96,165,250,0.05));
}

.hero-card h2 {
    margin: 0 0 8px;
    font-size: 1.08rem;
    letter-spacing: -0.02em;
}

.hero-card p {
    margin: 0;
    color: var(--text-soft);
    line-height: 1.55;
    font-size: 0.9rem;
}

.app-grid {
    align-items: flex-start !important;
    gap: 18px !important;
}

.main-panel,
.side-panel {
    min-width: 0 !important;
}

.side-panel {
    position: sticky;
    top: 18px;
}

.quickbar {
    display: flex;
    gap: 9px;
    flex-wrap: wrap;
    margin: 0 0 14px;
}

.quick-btn {
    border: 1px solid var(--border);
    color: var(--text-soft);
    background: rgba(255,255,255,0.045);
    border-radius: 999px;
    padding: 9px 13px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: 180ms ease;
}

.quick-btn:hover {
    color: white;
    border-color: var(--border-strong);
    background: rgba(16, 185, 129, 0.12);
    transform: translateY(-1px);
}

#chatbot {
    border: 1px solid var(--border) !important;
    background: linear-gradient(180deg, rgba(15,23,42,0.74), rgba(8,13,24,0.94)) !important;
    border-radius: 26px !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}

#chatbot .message,
#chatbot .bubble {
    border-radius: 18px !important;
    line-height: 1.55 !important;
}

.input-card {
    margin-top: 14px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 26px;
    background: rgba(5, 11, 20, 0.78);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
}

.input-card:focus-within {
    border-color: var(--border-strong);
    box-shadow: 0 0 0 4px rgba(16,185,129,0.10), 0 18px 50px rgba(0,0,0,0.28);
}

#msg-input,
#msg-input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    line-height: 1.65 !important;
}

#msg-input textarea {
    min-height: 86px !important;
    padding: 6px 8px 8px !important;
}

#msg-input textarea::placeholder {
    color: var(--muted) !important;
}

.input-actions {
    display: flex;
    align-items: center;
    gap: 10px;
    border-top: 1px solid var(--border);
    padding-top: 11px;
}

.input-hint {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-right: auto;
}

#send-btn,
#clear-btn {
    border-radius: 14px !important;
    font-weight: 800 !important;
    min-height: 42px !important;
    border: none !important;
    transition: 180ms ease !important;
}

#send-btn {
    background: linear-gradient(135deg, #10b981, #22c55e) !important;
    color: #03140e !important;
    box-shadow: 0 12px 28px rgba(16,185,129,0.28) !important;
}

#send-btn:hover,
#clear-btn:hover {
    transform: translateY(-1px) !important;
}

#clear-btn {
    background: rgba(255,255,255,0.07) !important;
    color: var(--text-soft) !important;
    border: 1px solid var(--border) !important;
}

.side-card {
    border: 1px solid var(--border);
    border-radius: 24px;
    background: var(--card);
    backdrop-filter: blur(16px);
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.23);
}

.card-title {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 12px;
}

#mood-label,
#session-stats,
#tool-log {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#mood-label input,
#mood-label textarea {
    background: transparent !important;
    color: var(--green) !important;
    border: none !important;
    text-align: center !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    padding: 5px 0 !important;
}

#session-stats textarea,
#tool-log textarea {
    background: transparent !important;
    color: var(--text-soft) !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    line-height: 1.75 !important;
    padding: 0 !important;
}

.risk-badge {
    padding: 14px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.045);
}

.risk-badge > div {
    display: flex;
    align-items: center;
    gap: 9px;
}

.risk-badge strong {
    font-size: 0.96rem;
}

.risk-badge small {
    display: block;
    margin-top: 6px;
    color: var(--muted);
    font-size: 0.75rem;
}

.risk-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: currentColor;
    box-shadow: 0 0 18px currentColor;
}

.risk-none {
    color: var(--green);
}

.risk-low {
    color: var(--blue);
}

.risk-medium {
    color: var(--amber);
}

.risk-high {
    color: var(--red);
}

.disclaimer {
    color: var(--text-soft);
    line-height: 1.62;
    font-size: 0.82rem;
}

.disclaimer strong {
    color: var(--green);
}

.disclaimer .warn {
    color: var(--amber);
}

.metric-list {
    display: grid;
    gap: 8px;
}

.metric {
    display: flex;
    justify-content: space-between;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 15px;
    background: rgba(255,255,255,0.04);
    color: var(--text-soft);
    font-size: 0.8rem;
}

.metric b {
    color: var(--text);
}

.ss-footer {
    margin-top: 22px;
    text-align: center;
    color: var(--muted-2);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.12em;
    font-size: 0.68rem;
}

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.22);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(148,163,184,0.35);
}

@media (max-width: 980px) {
    .gradio-container {
        padding: 16px 14px 28px !important;
    }

    .ss-header {
        grid-template-columns: 1fr;
    }

    .status-strip {
        justify-content: flex-start;
    }

    .side-panel {
        position: static;
    }

    .input-actions {
        flex-wrap: wrap;
    }

    .input-hint {
        width: 100%;
    }
}

@media (max-width: 640px) {
    .brand-row {
        align-items: flex-start;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 15px;
    }

    .quickbar {
        overflow-x: auto;
        flex-wrap: nowrap;
        padding-bottom: 4px;
    }

    .quick-btn {
        white-space: nowrap;
    }

    #chatbot {
        height: 440px !important;
    }
}
"""


with gr.Blocks(title="SafeSpace AI") as demo:
    gr.HTML('<div class="ss-shell">')

    gr.HTML(
        """
        <header class="ss-header">
            <div class="brand-row">
                <div class="brand-icon">🌿</div>
                <div>
                    <h1 class="brand-title">Safe<span>Space</span> AI</h1>
                    <div class="brand-subtitle">
                        A calm, guided support interface for mental-health conversations and medical assistance.
                    </div>
                </div>
            </div>

            <div class="status-strip">
                <span class="status-pill pill-green">MedGemma</span>
                <span class="status-pill pill-blue">Llama 3.2</span>
                <span class="status-pill pill-red">Emergency Ready</span>
            </div>
        </header>
        """
    )

    with gr.Row(elem_classes="app-grid"):
        with gr.Column(scale=8, elem_classes="main-panel"):
            gr.HTML(
                """
                <section class="hero-card">
                    <h2>How are you feeling right now?</h2>
                    <p>
                        Use a quick prompt or write freely. The assistant will detect mood,
                        call the right agent, and show safety status in real time.
                    </p>
                </section>
                """
            )

            gr.HTML(
                """
                <div class="quickbar">
                    <button class="quick-btn" onclick="setMsg('I am feeling anxious today')">😟 Feeling anxious</button>
                    <button class="quick-btn" onclick="setMsg('Find hospitals in Hof')">🏥 Find hospitals</button>
                    <button class="quick-btn" onclick="setMsg('Find pharmacy near me')">💊 Find pharmacy</button>
                    <button class="quick-btn" onclick="setMsg('I am having trouble sleeping')">😴 Sleep issues</button>
                    <button class="quick-btn" onclick="setMsg('I feel overwhelmed')">🌊 Overwhelmed</button>
                    <button class="quick-btn" onclick="setMsg('Give me a breathing exercise')">🫁 Breathing</button>
                    <button class="quick-btn" onclick="setMsg('Show my mood history')">📊 Mood history</button>
                    <button class="quick-btn" onclick="setMsg('Give me crisis hotline numbers')">📞 Hotlines</button>
                </div>

                <script>
                function setMsg(text) {
                    const ta = document.querySelector('#msg-input textarea');
                    if (!ta) return;

                    const nativeSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype,
                        'value'
                    ).set;

                    nativeSetter.call(ta, text);
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.focus();
                }
                </script>
                """
            )

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                height=540,
                layout="bubble",
                placeholder=(
                    "<div style='text-align:center;padding:110px 24px;'>"
                    "<div style='font-family:Playfair Display,serif;font-size:1.8rem;color:#dff7ef;'>"
                    "What's on your mind today?"
                    "</div>"
                    "<div style='font-size:0.78rem;color:#77879f;letter-spacing:0.16em;"
                    "text-transform:uppercase;margin-top:14px;font-family:JetBrains Mono,monospace;'>"
                    "safe · private · supportive"
                    "</div>"
                    "</div>"
                ),
            )

            with gr.Group(elem_classes="input-card"):
                msg = gr.Textbox(
                    placeholder="Share what's on your mind... I'm here to listen.",
                    show_label=False,
                    elem_id="msg-input",
                    lines=3,
                    max_lines=8,
                )

                with gr.Row(elem_classes="input-actions"):
                    gr.HTML(
                        '<span class="input-hint">Press Enter to send · Shift + Enter for new line</span>'
                    )
                    clear_btn = gr.Button("New session", elem_id="clear-btn", scale=1)
                    send_btn = gr.Button("Send →", elem_id="send-btn", scale=2)

        with gr.Column(scale=3, elem_classes="side-panel"):
            gr.HTML('<div class="side-card"><div class="card-title">Current Mood</div>')

            mood_label = gr.Textbox(
                value="😐 Neutral",
                elem_id="mood-label",
                show_label=False,
                interactive=False,
                lines=1,
            )

            gr.HTML("</div>")

            gr.HTML('<div class="side-card"><div class="card-title">Risk Status</div>')

            risk_status = gr.HTML(value=make_risk_badge("NONE"))

            gr.HTML("</div>")

            gr.HTML('<div class="side-card"><div class="card-title">Session Stats</div>')

            session_stats = gr.Textbox(
                value="Messages: 0\nRisk: NONE\nAgent: None",
                elem_id="session-stats",
                show_label=False,
                interactive=False,
                lines=3,
            )

            gr.HTML("</div>")

            gr.HTML(
                """
                <div class="side-card">
                    <div class="card-title">Safety Notice</div>
                    <div class="disclaimer">
                        SafeSpace AI is <span class="warn">not a substitute</span>
                        for professional medical care.<br><br>
                        In emergencies call <strong>112</strong> in Europe,
                        <strong>911</strong> in the USA, or <strong>999</strong> in the UK.
                    </div>
                </div>
                """
            )

            gr.HTML(
                """
                <div class="side-card">
                    <div class="card-title">System</div>
                    <div class="metric-list">
                        <div class="metric"><span>Backend</span><b>localhost:8000</b></div>
                        <div class="metric"><span>UI</span><b>Gradio</b></div>
                        <div class="metric"><span>Mode</span><b>Private Demo</b></div>
                    </div>
                </div>
                """
            )

            gr.HTML('<div class="side-card"><div class="card-title">Agent Log</div>')

            tool_log = gr.Textbox(
                elem_id="tool-log",
                show_label=False,
                lines=7,
                interactive=False,
                placeholder="Activity log will appear here...",
            )

            gr.HTML("</div>")

    gr.HTML(
        '<div class="ss-footer">SAFESPACE AI · MEDGEMMA + LLAMA 3.2 · CALM PROFESSIONAL UI</div>'
    )

    gr.HTML("</div>")

    history_state = gr.State([])
    tool_log_state = gr.State("")

    send_event = send_btn.click(
        fn=chat,
        inputs=[
            msg,
            history_state,
            tool_log_state,
            mood_label,
            session_stats,
            risk_status,
        ],
        outputs=[
            chatbot,
            msg,
            tool_log,
            mood_label,
            session_stats,
            risk_status,
        ],
    )

    send_event.then(
        fn=lambda h, t: (h, t),
        inputs=[chatbot, tool_log],
        outputs=[history_state, tool_log_state],
    )

    submit_event = msg.submit(
        fn=chat,
        inputs=[
            msg,
            history_state,
            tool_log_state,
            mood_label,
            session_stats,
            risk_status,
        ],
        outputs=[
            chatbot,
            msg,
            tool_log,
            mood_label,
            session_stats,
            risk_status,
        ],
    )

    submit_event.then(
        fn=lambda h, t: (h, t),
        inputs=[chatbot, tool_log],
        outputs=[history_state, tool_log_state],
    )

    clear_event = clear_btn.click(
        fn=clear_chat,
        outputs=[
            chatbot,
            msg,
            tool_log,
            mood_label,
            session_stats,
            risk_status,
        ],
    )

    clear_event.then(
        fn=lambda: ([], ""),
        outputs=[history_state, tool_log_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=False,
        css=CSS,
        theme=gr.themes.Base(),
    )