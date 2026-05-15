import streamlit as st
import requests
import base64
from pathlib import Path

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="College and Hostel Enquiry Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://127.0.0.1:8000/chat"
LOGO_PATH = Path("assets/footer-logo-icon.png")

# -------------------- HELPERS --------------------
def get_base64_image(image_path: Path):
    if image_path.exists():
        return base64.b64encode(image_path.read_bytes()).decode()
    return None

def format_chat_history_for_context(history, limit=4):
    if not history:
        return ""
    recent = history[-limit:]
    lines = []
    for item in recent:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Bot: {a}")
    return "\n".join(lines)

logo_base64 = get_base64_image(LOGO_PATH)

# -------------------- SESSION STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

if "use_memory" not in st.session_state:
    st.session_state.use_memory = True

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

if "show_panel" not in st.session_state:
    st.session_state.show_panel = True

# -------------------- THEME --------------------
if st.session_state.theme_mode == "Dark":
    bg1 = "#08111f"
    bg2 = "#0f172a"
    card = "rgba(17, 24, 39, 0.78)"
    border = "rgba(148, 163, 184, 0.22)"
    text = "#f8fafc"
    muted = "#cbd5e1"
    user_bg = "linear-gradient(135deg, #38bdf8, #2563eb)"
    bot_bg = "rgba(17, 24, 39, 0.96)"
    input_bg = "#0f172a"
    input_border = "#38bdf8"
    placeholder = "#94a3b8"
    info_bg = "rgba(30, 64, 175, 0.18)"
    info_text = "#93c5fd"
    memory_bg = "rgba(30, 41, 59, 0.78)"
else:
    bg1 = "#eef2ff"
    bg2 = "#f8fafc"
    card = "rgba(255, 255, 255, 0.92)"
    border = "rgba(15, 23, 42, 0.10)"
    text = "#0f172a"
    muted = "#475569"
    user_bg = "linear-gradient(135deg, #38bdf8, #2563eb)"
    bot_bg = "rgba(255, 255, 255, 0.98)"
    input_bg = "#ffffff"
    input_border = "#cbd5e1"
    placeholder = "#64748b"
    info_bg = "rgba(59, 130, 246, 0.10)"
    info_text = "#1d4ed8"
    memory_bg = "rgba(255, 255, 255, 0.88)"

# -------------------- CSS --------------------
st.markdown(
    f"""
<style>
header {{
    visibility: hidden;
}}
[data-testid="stToolbar"] {{
    display: none !important;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}
#MainMenu {{
    visibility: hidden;
}}
footer {{
    visibility: hidden;
}}
section[data-testid="stSidebar"] {{
    display: none !important;
}}

html, body, [class*="css"] {{
    font-family: "Segoe UI", sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.08), transparent 28%),
        linear-gradient(180deg, {bg1} 0%, {bg2} 100%);
    color: {text};
}}

.block-container {{
    max-width: 1400px;
    padding-top: 1rem !important;
    padding-bottom: 2rem;
}}

div:empty {{
    display: none !important;
}}

.app-card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.06);
}}

.header-card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 24px;
    padding: 18px 22px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 26px rgba(0,0,0,0.05);
}}

.logo-title-wrap {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.logo-round {{
    width: 74px;
    height: 74px;
    border-radius: 18px;
    border: 1px solid {border};
    object-fit: cover;
    background: white;
    padding: 6px;
}}

.main-title {{
    font-size: 2.2rem;
    font-weight: 800;
    color: {text};
    margin-bottom: 0.1rem;
}}

.sub-title {{
    color: {muted};
    font-size: 1rem;
}}

.section-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {text};
    margin-bottom: 10px;
}}

.small-muted {{
    color: {muted};
    font-size: 0.92rem;
}}

.user-wrap, .bot-wrap {{
    display: flex;
    margin: 16px 0;
}}

.user-wrap {{
    justify-content: flex-end;
}}

.bot-wrap {{
    justify-content: flex-start;
}}

.user-msg {{
    max-width: 78%;
    background: {user_bg};
    color: white;
    padding: 14px 16px;
    border-radius: 22px 22px 6px 22px;
}}

.bot-msg {{
    max-width: 78%;
    background: {bot_bg};
    color: {text};
    padding: 14px 16px;
    border-radius: 22px 22px 22px 6px;
    border: 1px solid {border};
}}

.msg-label {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {muted};
    margin-bottom: 6px;
}}

.memory-box {{
    background: {memory_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 12px 14px;
    margin-bottom: 1rem;
}}

.info-banner {{
    background: {info_bg};
    color: {info_text};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 1rem;
    font-weight: 600;
}}

.stTextInput > div > div > input {{
    background-color: {input_bg} !important;
    color: {text} !important;
    border: 2px solid {input_border} !important;
    border-radius: 16px !important;
    padding: 0.95rem 1rem !important;
    font-size: 1rem !important;
}}

.stTextInput > div > div > input::placeholder {{
    color: {placeholder} !important;
    opacity: 1 !important;
}}

.stTextInput > label {{
    color: {muted} !important;
    font-weight: 600 !important;
}}

.stButton button,
button[kind="secondaryFormSubmit"] {{
    background: linear-gradient(135deg, #38bdf8, #2563eb) !important;
    color: white !important;
    border: 1px solid #2563eb !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    min-height: 48px !important;
    width: 100%;
}}

.stButton button p,
button[kind="secondaryFormSubmit"] p {{
    color: white !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 14px !important;
}}
</style>
""",
    unsafe_allow_html=True
)

# -------------------- TOP BAR --------------------
top_col1, top_col2, _ = st.columns([1.4, 1.4, 6])

with top_col1:
    if st.button("☰ Hide Panel" if st.session_state.show_panel else "☰ Show Panel"):
        st.session_state.show_panel = not st.session_state.show_panel
        st.rerun()

with top_col2:
    theme_label = "🌙 Dark Mode" if st.session_state.theme_mode == "Light" else "☀ Light Mode"
    if st.button(theme_label):
        st.session_state.theme_mode = "Dark" if st.session_state.theme_mode == "Light" else "Light"
        st.rerun()

# -------------------- LAYOUT --------------------
if st.session_state.show_panel:
    left_col, right_col = st.columns([1.05, 3], gap="large")
else:
    left_col = None
    right_col = st.container()

# -------------------- LEFT PANEL --------------------
if st.session_state.show_panel:
    with left_col:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)

        if logo_base64:
            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom:14px;">
                    <img src="data:image/png;base64,{logo_base64}" width="100"
                         style="border-radius:16px; background:white; padding:6px; border:1px solid {border};" />
                    <div style="margin-top:10px; color:{text}; font-size:1.05rem; font-weight:800;">
                        College and Hostel Enquiry Chatbot
                    </div>
                    <div class="small-muted" style="margin-top:8px;">
                        Smart assistant for college, hostel, admissions, departments and facilities
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state.use_memory = st.toggle(
            "Use chat memory context",
            value=st.session_state.use_memory,
            key="memory_toggle_left"
        )

        st.markdown(
            '<div class="section-title" style="margin-top:18px;">Suggested Questions</div>',
            unsafe_allow_html=True
        )

        suggestions = [
            "Where is GHEC located?",
            "What is the name of the college?",
            "Is hostel facility available?",
            "Who is the HOD of CSE?",
            "What facilities are available in hostel?",
            "When was GHEC established?",
            "Is the college government or private?",
            "What courses are offered?"
        ]

        for i, q in enumerate(suggestions):
            if st.button(q, key=f"suggestion_{i}"):
                st.session_state.selected_question = q
                st.rerun()

        st.markdown("---")

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.selected_question = ""
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- RIGHT MAIN --------------------
with right_col:
    if logo_base64:
        st.markdown(
            f"""
            <div class="header-card">
                <div class="logo-title-wrap">
                    <img class="logo-round" src="data:image/png;base64,{logo_base64}" />
                    <div>
                        <div class="main-title">College and Hostel Enquiry Chatbot</div>
                        <div class="sub-title">Ask anything about the college, hostel, admissions, departments and facilities</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="header-card">
                <div class="main-title">🎓 College and Hostel Enquiry Chatbot</div>
                <div class="sub-title">Ask anything about the college, hostel, admissions, departments and facilities</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.use_memory and st.session_state.chat_history:
        memory_context = format_chat_history_for_context(st.session_state.chat_history, limit=3)
        st.markdown(
            f"""
            <div class="memory-box">
                <b>Chat Memory Context</b><br><br>
                <pre style="white-space: pre-wrap; margin:0; font-family:Segoe UI; color:{text};">{memory_context}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="info-banner">Start by asking a question from the input box below.</div>',
            unsafe_allow_html=True
        )
    else:
        for chat in st.session_state.chat_history:
            st.markdown(
                f"""
                <div class="user-wrap">
                    <div class="user-msg">
                        <div class="msg-label">You</div>
                        {chat["question"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="bot-wrap">
                    <div class="bot-msg">
                        <div class="msg-label">Bot</div>
                        {chat["answer"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    default_question = st.session_state.get("selected_question", "")

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([8, 1])

        with c1:
            user_input = st.text_input(
                "Ask your question",
                value=default_question,
                placeholder="Type your question here..."
            )

        with c2:
            submitted = st.form_submit_button("Ask")

    if submitted:
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Fetching answer..."):
                try:
                    payload = {"question": user_input}

                    if st.session_state.use_memory:
                        payload["chat_history"] = st.session_state.chat_history[-4:]

                    response = requests.post(API_URL, json=payload, timeout=90)
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")

                    if (
                        not st.session_state.chat_history
                        or st.session_state.chat_history[-1]["question"] != user_input
                    ):
                        st.session_state.chat_history.append({
                            "question": user_input,
                            "answer": answer
                        })

                    st.session_state.selected_question = ""
                    st.rerun()

                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")