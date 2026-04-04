import streamlit as st
import pandas as pd
import cv2
import numpy as np
from database import conn, cursor
from recomsystem import generate_recommendation

def generate_smart_tip(skin, occasion, weather):
    
    if occasion == "Party":
        return "✨ Go bold! Dark shades + statement makeup will enhance your look."
    
    elif weather == "Hot":
        return "🌞 Use light fabrics & minimal makeup to stay fresh."
    
    elif skin == "dark":
        return "🔥 Bright colors like yellow & white will pop beautifully on you."
    
    elif skin == "light":
        return "🌸 Soft pastels and pink tones will enhance your glow."
    
    else:
        return "✨ Keep it balanced and stylish!"

skin_color_map = {
    "fair": {
        "best": ["blue", "red", "black"],
        "good": ["pink", "purple", "navy"],
        "avoid": ["yellow"]
    },
    "medium": {
        "best": ["green", "white", "navy"],
        "good": ["blue", "maroon", "black"],
        "avoid": ["beige"]
    },
    "dark": {
        "best": ["yellow", "white", "pink"],
        "good": ["red", "orange", "blue"],
        "avoid": ["brown"]
    }
}

def suggest_color(skin_tone, available_colors):
    if skin_tone not in skin_color_map:
        return "No skin tone match found"

    preferences = skin_color_map[skin_tone]

    best_match = []
    good_match = []

    for color in available_colors:
        if color in preferences["best"]:
            best_match.append(color)
        elif color in preferences["good"]:
            good_match.append(color)

    # Priority logic
    if best_match:
        return f"🔥 Best choice: {best_match[0]}"
    elif good_match:
        return f"👍 Good choice: {good_match[0]}"
    else:
        return "⚠️ Try neutral colors like black or white"

st.markdown("""
<style>

/* 🌌 GLOBAL BACKGROUND (ALL PAGES SAME) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b, #ec4899);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

/* ANIMATION */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AuraStyle AI", layout="wide")

def extract_color(word):
    mapping = {
        "red": "red",
        "pink": "pink",
        "coral": "coral",
        "nude": "#d2a679",
        "rose": "#ff66cc",
        "berry": "#800040",
        "wine": "#722f37",
        "brown": "#8b4513",
        "peach": "#ffb07c",
        "blue": "blue",
        "navy": "navy",
        "royal": "#4169e1",   
        "magenta": "magenta",
        "purple": "purple",
        "grey": "grey"
    }

    word = word.lower()

    for key in mapping:
        if key in word:
            return mapping[key]

    return "#cccccc"

def show_color_palette(colors_string):
    colors = [c.strip() for c in colors_string.split(",") if c.strip()]

    st.subheader("🎨 Your Color Palette")

    cols = st.columns(len(colors))

    for i, color in enumerate(colors):
        with cols[i]:
            safe_color = extract_color(color)  

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {safe_color}, black);
                    height:90px;
                    border-radius:12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                "></div>
                <p style="text-align:center; margin-top:5px;">{color}</p>
                """,
                unsafe_allow_html=True
            )

def show_makeup_visuals(lipstick, blush):

    lip_color = extract_color(lipstick)

    blush_color = extract_color(blush)
    if "not available" in blush.lower():
       blush_color = "#ff9aa2"   # soft pink fallback

    st.subheader("💄 Makeup Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color:{lip_color};
                height:80px;
                width:80px;
                border-radius:50%;
                margin:auto;
            "></div>
            <p style="text-align:center; font-weight:bold;">💄 Lipstick</p>
            <p style="text-align:center;">{lipstick}</p>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color:{blush_color};
                height:80px;
                width:80px;
                border-radius:50%;
                margin:auto;
            "></div>
            <p style="text-align:center; font-weight:bold;">🌸 Blush</p>
            <p style="text-align:center;">{blush}</p>
            """,
            unsafe_allow_html=True
        )

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = ""
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# ---------------- LOAD DATASETS GLOBALLY ----------------
# ✅ Loading datasets here ONCE makes your app blazing fast!
skin_df = pd.read_csv("Skin_dataset.csv")
skin_df.columns = skin_df.columns.str.strip().str.lower().str.replace(" ", "_")

def load_and_clean_df(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
    return df

rec_df = load_and_clean_df("recommendation_dataset.csv")
makeup_df = load_and_clean_df("makeup_dataset.csv")
brand_df = load_and_clean_df("Brand_dataset.csv")


# ---------------- AI PREDICTION ----------------
def predict_skin_from_dataset(h, s, v):
    
    df = skin_df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["distance"] = (
        (df["hue"] - h) ** 2 +
        (df["saturation"] - s) ** 2 +
        (df["value"] - v) ** 2
    )

    nearest = df.loc[df["distance"].idxmin()]

    if v > 180:
        skin = "light"
    elif v > 130:
       skin = "medium"
    else:
        skin = "dark"

    if h < 20 and s > 100:
     undertone = "warm"
    elif h < 35:
     undertone = "neutral"
    else:
     undertone = "cool"

    return skin, undertone

# ---------------- HSV EXTRACTION ----------------
def extract_hsv_from_image(image_data):

    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        face = img
    else:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))

    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

    h = int(np.mean(hsv[:, :, 0]))
    s = int(np.mean(hsv[:, :, 1]))
    v = int(np.mean(hsv[:, :, 2]))

    return h, s, v

def get_base64_bin(file_path):
    with open(file_path, "rb") as f:
        import base64
        return base64.b64encode(f.read()).decode()

# ---------------- LOGIN PAGE ----------------
def login_page():

    st.markdown("""
    <style>

    /* Hide header */
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}

    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px;
    }

    
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* 💎 TITLE */
    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        color: white;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }

    .sub-title {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 40px;
        font-style: italic;
    }

    /* 🧊 GLASS LOGIN CARD */
    .login-card {
        background: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }

    /* INPUT FIELDS */
    div[data-baseweb="input"] > div {
        background-color: #1e293b !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    div[data-baseweb="input"] input {
        color: #ffffff !important;
    }

    /* BUTTON */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 5px 20px rgba(0,114,255,0.5);
    }

    /* LABEL COLORS */
    .stRadio label, .stTextInput label {
        color: #e2e8f0 !important;
    }

    /* 🎬 GIF SIZE CONTROL */
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .video-box {
        width: 70%;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }

    </style>
    """, unsafe_allow_html=True)

    # Titles
    st.markdown("<h1 class='main-title'>👗 AuraStyle AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your Digital Fashion Stylist</p>", unsafe_allow_html=True)

    # Layout
    _, col1, col2, _ = st.columns([0.1, 1, 1, 0.1])

    # 🎬 GIF (SMALL + CENTERED)
    with col1:
        try:
            bin_str = get_base64_bin("GIF.mp4")
            st.markdown(
                f"""
                <div class="video-container">
                    <video autoplay loop muted playsinline class="video-box">
                        <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
                    </video>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"GIF not found: {e}")

    # 🔐 LOGIN CARD
    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        st.markdown("### ✨Your Personsal Stylist Awaits")

        option = st.radio("Mode", ["Login", "Sign Up"], horizontal=True)

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if option == "Sign Up":
            if st.button("CREATE ACCOUNT 🚀", use_container_width=True):
                cursor.execute("SELECT * FROM users WHERE email=?", (email,))
                if cursor.fetchone():
                    st.warning("User already exists!")
                else:
                    cursor.execute("INSERT INTO users VALUES (?,?)", (email, password))
                    conn.commit()
                    st.success("Account created! Please login.")
        else:
            if st.button("LOGIN 🚀", use_container_width=True):
                cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
                if cursor.fetchone():
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
def dashboard():

    user = st.session_state.email.split("@")[0].capitalize()

    st.sidebar.markdown(f"### 💎 Stylist: {user}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("👗 Style Dashboard")
    st.markdown(f"### Welcome back, **{user}** ✨")

    st.divider()

    col1, col2 = st.columns(2)

    # USER INPUT
    with col1:
        Name = st.text_input("Name")
        Age = st.slider("Age", 10, 80)

        Occasion = st.selectbox("Occasion",
            ["Casual", "Office", "Wedding", "Party", "Date", "Festival"])

        Weather = st.selectbox("Weather",
            ["Hot", "Cold", "Rainy", "Pleasant"])
        color_input = st.text_input("Enter colors you have (comma separated):")
        available_colors = []

        if color_input:
          available_colors = [c.strip().lower() for c in color_input.split(",")]

        # ✅ Using the global brand_df here instead of reading the CSV again
        brand_list = sorted(brand_df["brand"].str.title().unique())
        brand = st.selectbox("Favorite Brand", brand_list)

    # IMAGE INPUT
    with col2:
        method = st.radio("Image Input", ["Camera", "Upload"])

        image_data = None
        if method == "Camera":
            img = st.camera_input("Take Photo")
            if img:
                st.image(img, width=250)
                image_data = img
        else:
            file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
            if file:
                st.image(file, width=250)
                image_data = file

    st.divider()

    if st.button("✅ OK - Analyze My Face"):

        if not Name:
            st.warning("Please enter your name")
            return

        if image_data is None:
            st.warning("Please upload/capture your image")
            return

        try:
            image_data.seek(0)
        except:
            pass

        h, s, v = extract_hsv_from_image(image_data)

        skin, undertone = predict_skin_from_dataset(h, s, v)

        st.session_state.page = "result"
        st.session_state.user_name = Name
        st.session_state.skin_tone = skin
        st.session_state.undertone = undertone
        st.session_state.occasion = Occasion
        st.session_state.weather = Weather
        st.session_state.brand = brand
        st.session_state.available_colors = available_colors
        st.rerun()
        # ✅ after skin_tone is calculated

# ---------------- RESULT PAGE ----------------
def result_page():

    name = st.session_state.get("user_name")
    skin = st.session_state.get("skin_tone")
    undertone = st.session_state.get("undertone")
    occasion = st.session_state.get("occasion")
    weather = st.session_state.get("weather")
    brand = st.session_state.get("brand")

    # ✅ SAFETY CHECK (VERY IMPORTANT)
    if not all([skin, undertone, occasion, weather, brand]):
        st.error("Missing user data. Please go back and try again.")
        return

    # ✅ DATA LOAD
    rec_df = load_and_clean_df("Recommendation_dataset.csv")
    makeup_df = load_and_clean_df("Makeup_dataset.csv")
    brand_df = load_and_clean_df("Brand_dataset.csv")

    # ✅ GENERATE RESULT
    result = generate_recommendation(
        skin, undertone, occasion, weather, brand,
        rec_df, makeup_df, brand_df
    )

    # ✅ ERROR CHECK
    if result is None:
        st.error("Something went wrong. Please try again.")
        return

    # ---------------- UI ----------------
    st.title(f"✨ Hey {name}!")

    # ---------------- SKIN ----------------
    st.subheader("🧠 Skin Analysis")
    st.success(f"Your skin tone is **{skin} ({undertone})**")

    compliments = {
        "light": "Your skin has a soft radiant glow 🌸",
        "medium": "Your skin tone is rich & versatile 🔥",
        "dark": "Your skin tone is bold & stunning ✨"
    }

    st.info(compliments.get(skin, "Your skin tone is beautiful! ✨"))
    st.info(f"✨ These shades complement your {skin} ({undertone}) tone beautifully.")

    # ✅ highlight main color
    st.success(f"✨ Recommended color: {result.get('color', 'black')}")
    available_colors = st.session_state.get("available_colors", [])

    if available_colors:
        suggested = suggest_color(skin, available_colors)

        st.subheader("🧠 Smart Wardrobe Suggestion")
        st.success(f"✨ {suggested}")
    else:
        st.warning("⚠ You didn't enter your available colors")
    # 🎯 USER COLOR MATCHING AI


    st.divider()

    col1, col2 = st.columns(2)

    # ---------------- LEFT ----------------
    with col1:
        show_color_palette(result.get("all_colors", ""))

        avoid = result.get("avoid", "")
        avoid_list = [c.strip() for c in avoid.split(",") if c.strip()]

        if avoid_list:
            st.markdown("🚫 **Avoid:** " + ", ".join(avoid_list))

    # ---------------- RIGHT ----------------
    with col2:
        st.subheader("💄 Makeup Kit")

        foundation = result.get("foundation", "No product found")
        # 🔍 DEBUG CHECK
        st.write("Lipstick from dataset:", result.get("lipstick"))
        st.write("Blush from dataset:", result.get("blush"))

        if foundation == "No product found":
            st.warning("⚠ No perfect foundation match found")
        else:
            st.success(f"🧴 Foundation: {foundation}")

        lipstick = result.get("lipstick", "No product found")
        blush = result.get("blush", "No product found")

        if blush == "No product found":
            blush = "Matching shade not available"

        show_makeup_visuals(lipstick, blush)

    # ---------------- TIP ----------------
    st.divider()
    st.subheader("💡 Personalized Tip")
    tip = generate_smart_tip(skin, occasion, weather)
    st.success(tip)

    if st.button("🔙 Back"):
        st.session_state.page = "dashboard"
        st.rerun()

# ---------------- RUN ----------------
if st.session_state.logged_in:
    if st.session_state.page == "dashboard":
        dashboard()
    else:
        result_page()
else:
    login_page()