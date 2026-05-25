import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Email & SMS Spam Detector",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    cv = CountVectorizer()
    X = cv.fit_transform(df["message"])

    model = MultinomialNB()
    model.fit(X, df["label"])

    return model, cv

model, cv = load_model()

# ---------------- SESSION ----------------
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- STYLE ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#020617,#081b3d,#0a2247);
    color:white;
}

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#071938,#020617);
    border-right:1px solid rgba(0,229,255,0.18);
}

.menu-title{
    text-align:center;
    font-size:30px;
    font-weight:800;
    color:white;
    text-shadow:0 0 12px #00e5ff;
}

.menu-card{
    background:rgba(255,255,255,0.04);
    border-radius:18px;
    padding:18px;
    border:1px solid rgba(0,229,255,0.15);
    box-shadow:0 0 20px rgba(0,229,255,0.10);
}

.member-name{
    font-size:28px;
    font-weight:900;
    margin-bottom:8px;
    background: linear-gradient(90deg,#FFD700,#ff3b3b);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow:
        0 0 12px rgba(255,215,0,0.65),
        0 0 24px rgba(255,59,59,0.35);
}

.member-role{
    color:#dbeafe;
    font-size:15px;
    margin-bottom:18px;
}

.main-title{
    text-align:center;
    font-size:64px;
    font-weight:900;
    color:white;
    text-shadow:
        0 0 18px #00e5ff,
        0 0 35px #ff00ff;
}

textarea{
    border:2px solid #00e5ff !important;
    border-radius:18px !important;
    min-height:180px !important;
    font-size:18px !important;
    background:rgba(255,255,255,0.03) !important;
    color:white !important;
}

div.stButton > button{
    background:linear-gradient(90deg,#00c6ff,#ff00ff);
    color:white;
    font-size:20px;
    font-weight:800;
    border:none;
    border-radius:14px;
    padding:12px 30px;
    box-shadow:0 0 20px #00e5ff;
}

div[role="radiogroup"] label{
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- WAVES ----------------
components.html("""
<div style="
position:fixed;
bottom:0;
left:0;
width:100%;
overflow:hidden;
line-height:0;
z-index:-1;
">
<svg viewBox="0 0 1440 320"
style="width:200%;height:220px;animation:waveMove 10s linear infinite;">
<path fill="#00e5ff" fill-opacity="0.16"
d="M0,192L48,176C96,160,192,128,288,128C384,128,480,160,576,181.3C672,203,768,213,864,202.7C960,192,1056,160,1152,154.7C1248,149,1344,171,1392,181.3L1440,192L1440,320L0,320Z"></path>
</svg>
</div>

<style>
@keyframes waveMove{
0%{transform:translateX(0);}
100%{transform:translateX(-50%);}
}
</style>
""", height=0)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("<div class='menu-title'>☰ Menu</div>", unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["👥 About Us", "📩 About Project", "🤖 Model Info"]
    )

    st.markdown("<div class='menu-card'>", unsafe_allow_html=True)

    if menu == "👥 About Us":
        st.markdown("""
        <div style="text-align:center;">

        <h2>🧑‍💻 Meet Our Team</h2>

        <div class="member-name">🔥 Shivaraj PM</div>
        <div class="member-role">Lead Developer • Project Design</div>

        <div class="member-name">⚡ Vijaykumar</div>
        <div class="member-role">Project Contributor • Development Support</div>

        <div class="member-name">🚀 Balu</div>
        <div class="member-role">Project Contributor • Testing & Support</div>

        <br>

        🤝 Built together with teamwork & Machine Learning

        </div>
        """, unsafe_allow_html=True)

    elif menu == "📩 About Project":
        st.markdown("""
### 📩 About Project

Email & SMS Spam Detector classifies messages into:

✅ **NOT SPAM**  
⚠️ **SPAM**

Detects unwanted or suspicious messages using Machine Learning.
""")

    elif menu == "🤖 Model Info":
        st.markdown("""
### 🤖 Model Info

**Model Used:** Multinomial Naive Bayes  

**Vectorizer:** CountVectorizer  

**Dataset:** spam.csv
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='main-title'>📩 Email & SMS Spam Detector</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------
message = st.text_area(
    "✍️ Enter your message",
    placeholder="Type your email or SMS here..."
)

# ---------------- CENTER BUTTON ----------------
left, center, right = st.columns([1,2,1])

with center:
    check = st.button(
        "🚀 Check Message",
        use_container_width=True
    )

# ---------------- PREDICT ----------------
if check:
    if message.strip():

        data = cv.transform([message])
        pred = model.predict(data)[0]

        if pred == "spam":
            st.session_state.result = "SPAM"
        else:
            st.session_state.result = "NOT SPAM"

# ---------------- POPUP ----------------
if st.session_state.result:

    @st.dialog("Prediction Result")
    def result_popup():

        if st.session_state.result == "SPAM":
            st.markdown(
                "<h1 style='text-align:center;color:#ff4b6e;'>⚠️ SPAM</h1>",
                unsafe_allow_html=True
            )
            st.error("Warning! Spam detected.")
        else:
            st.markdown(
                "<h1 style='text-align:center;color:#00ff99;'>✅ NOT SPAM</h1>",
                unsafe_allow_html=True
            )
            st.success("This message looks safe.")

        st.write("")

        c1, c2, c3 = st.columns([1,2,1])

        with c2:
            if st.button("❌ Close", use_container_width=True):
                st.session_state.result = None
                st.rerun()

    result_popup()
# ---------------- DISCLAIMER ----------------
st.markdown("""
<br><br>
<div style="
text-align:center;
padding:16px;
margin-top:30px;
margin-bottom:15px;
border-radius:14px;
background:rgba(255,255,255,0.04);
border:1px solid rgba(0,229,255,0.15);
color:#dbeafe;
font-size:14px;
">

⚠️ <b>Disclaimer:</b> This Email & SMS Spam Detector provides predictions using machine learning and has approximately
<b style='color:#00e5ff;'>96% accuracy</b>.

Results may occasionally be incorrect. Please use this tool wisely and verify important messages before taking action.

</div>
""", unsafe_allow_html=True)