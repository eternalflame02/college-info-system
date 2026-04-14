"""
MBCET CSE Department Chatbot — Streamlit UI.

Launch with:
    python main.py --stage chat
    # or directly:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBCET CSE Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Dark theme overrides ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* ── Chat messages ── */
.stChatMessage {
    border-radius: 16px !important;
    margin-bottom: 12px !important;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e6edf3 !important;
}

/* ── Source card ── */
.source-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: all 0.2s ease;
}
.source-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(99, 179, 237, 0.3);
}
.source-card .source-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #63b3ed;
    margin-bottom: 4px;
}
.source-card .source-text {
    font-size: 13px;
    color: #a0aec0;
    line-height: 1.5;
}

/* ── Badge styles ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-teaching { background: rgba(129, 230, 217, 0.15); color: #81e6d9; }
.badge-faculty  { background: rgba(99, 179, 237, 0.15); color: #63b3ed; }
.badge-course   { background: rgba(183, 148, 244, 0.15); color: #b794f4; }
.badge-regulation { background: rgba(252, 211, 77, 0.15); color: #fcd34d; }
.badge-timetable { background: rgba(251, 146, 60, 0.15); color: #fb923c; }
.badge-general  { background: rgba(160, 174, 192, 0.15); color: #a0aec0; }

.badge-excellent { background: rgba(72, 187, 120, 0.15); color: #48bb78; }
.badge-good      { background: rgba(99, 179, 237, 0.15); color: #63b3ed; }
.badge-fair      { background: rgba(252, 211, 77, 0.15); color: #fcd34d; }
.badge-poor      { background: rgba(245, 101, 101, 0.15); color: #f56565; }
.badge-none      { background: rgba(160, 174, 192, 0.15); color: #a0aec0; }

/* ── Hero title ── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #b794f4, #81e6d9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-subtitle {
    text-align: center;
    color: #718096;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    flex: 1;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3;
}
.metric-card .metric-label {
    font-size: 0.7rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Resource Loading (cached) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_chatbot():
    """Load chatbot resources once."""
    from chatbot import warmup, answer_question
    warmup()
    return answer_question


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎓 MBCET CSE Assistant")
    st.markdown("---")

    st.markdown("""
    **Ask me about:**
    - 👤 Faculty information
    - 📚 Courses & syllabi
    - 👨‍🏫 Who teaches what
    - 📋 Regulations & policies
    - 🕐 Timetables
    """)

    st.markdown("---")

    st.markdown("##### 💡 Example queries")
    examples = [
        "Who is the HOD of CSE?",
        "Who teaches Data Structures?",
        "What are the courses in Semester 5?",
        "What is the qualification of Dr. Jisha John?",
        "What is the attendance requirement?",
        "Tell me about the CSE department",
    ]
    for ex in examples:
        if st.button(f"→ {ex}", key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill_query"] = ex

    st.markdown("---")

    # Stats
    st.markdown("##### 📊 Knowledge Base")
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">2,203</div>
            <div class="metric-label">Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">43</div>
            <div class="metric-label">Faculty</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by EmbeddingGemma + ChromaDB + Groq (Llama 3.3)")


# ── Main Chat Area ───────────────────────────────────────────────────────

# Hero
st.markdown('<h1 class="hero-title">MBCET CSE Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Ask anything about the Computer Science & Engineering department</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 Sources ({len(message['sources'])} chunks)", expanded=False):
                for src in message["sources"]:
                    source_name = Path(src["source_file"]).stem if src.get("source_file") else "unknown"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">{src.get('content_type', '?')} · {source_name} · Distance: {src.get('distance', '?'):.3f}</div>
                        <div class="source-text">{src.get('text', '')[:300]}{'...' if len(src.get('text', '')) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Show metadata badges
        if message["role"] == "assistant" and message.get("meta"):
            meta = message["meta"]
            badges = []
            qt = meta.get("query_type", "general")
            badges.append(f'<span class="badge badge-{qt}">{qt}</span>')
            q = meta.get("quality", "unknown")
            badges.append(f'<span class="badge badge-{q}">Quality: {q}</span>')
            rt = meta.get("response_time_ms", 0)
            badges.append(f'<span class="badge badge-general">{rt:.0f}ms</span>')
            st.markdown(" ".join(badges), unsafe_allow_html=True)


# Handle prefill from sidebar
prefill = st.session_state.pop("prefill_query", None)

# Chat input
prompt = st.chat_input("Ask about MBCET CSE department...", key="chat_input")

# Use prefilled query if available
if prefill and not prompt:
    prompt = prefill

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_fn = load_chatbot()
            response = answer_fn(prompt)

        st.markdown(response.answer, unsafe_allow_html=True)

        # Prepare source data for storage
        sources = []
        for chunk in response.source_chunks[:5]:
            sources.append({
                "source_file": chunk.source_file,
                "content_type": chunk.content_type,
                "distance": chunk.distance,
                "text": chunk.text,
            })

        if sources:
            with st.expander(f"📎 Sources ({len(sources)} chunks)", expanded=False):
                for src in sources:
                    source_name = Path(src["source_file"]).stem if src.get("source_file") else "unknown"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">{src.get('content_type', '?')} · {source_name} · Distance: {src.get('distance', 0):.3f}</div>
                        <div class="source-text">{src.get('text', '')[:300]}{'...' if len(src.get('text', '')) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Show badges
        meta = {
            "query_type": response.query_type,
            "quality": response.quality,
            "response_time_ms": response.response_time_ms,
        }
        badges = []
        qt = meta["query_type"]
        badges.append(f'<span class="badge badge-{qt}">{qt}</span>')
        q = meta["quality"]
        badges.append(f'<span class="badge badge-{q}">Quality: {q}</span>')
        rt = meta["response_time_ms"]
        badges.append(f'<span class="badge badge-general">{rt:.0f}ms</span>')

        if response.kg_answer:
            badges.append(f'<span class="badge badge-teaching">KG Hit</span>')

        st.markdown(" ".join(badges), unsafe_allow_html=True)

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": sources,
        "meta": meta,
    })
