import pandas as pd
import random

# ---------------- LOAD + CLEAN ----------------
def load_clean_csv(path):
    df = pd.read_csv(path)
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Clean string data and prevent NaNs from becoming the word "nan"
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

    return df

# ---------------- PRE-LOAD DATASETS ----------------
# Load datasets once globally to prevent reading from disk on every function call.
# Uncomment and provide correct paths when ready to run.
# REC_DF = load_clean_csv("recommendation_dataset.csv")
# MAKEUP_DF = load_clean_csv("makeup_dataset.csv")
# BRAND_DF = load_clean_csv("brand_dataset.csv")


# ---------------- NORMALIZE INPUT ----------------
def normalize_inputs(occasion, weather, brand):
    occasion_map = {
        "casual": "casual",
        "office": "office",
        "wedding": "wedding",
        "party": "party",
        "date": "party",
        "festival": "wedding"
    }

    weather_map = {
        "hot": "summer",
        "cold": "winter",
        "rainy": "summer",
        "pleasant": "summer"
    }

    # Ensure inputs are lowered and stripped before mapping
    occasion_clean = occasion.lower().strip()
    weather_clean = weather.lower().strip()
    brand_clean = brand.lower().strip()

    return (
        occasion_map.get(occasion_clean, occasion_clean),
        weather_map.get(weather_clean, weather_clean),
        brand_clean
    )


# ---------------- SMART FILTER ----------------
def smart_filter(df, skin_tone, undertone, occasion, weather):
    # Cascading fallbacks
    exact = df[
        (df["skin_tone"] == skin_tone) &
        (df["undertone"] == undertone) &
        (df["occasion"] == occasion) &
        (df["weather"] == weather)
    ]
    if not exact.empty: return exact.sample(1)

    partial = df[
        (df["skin_tone"] == skin_tone) &
        (df["undertone"] == undertone) &
        (df["occasion"] == occasion)
    ]
    if not partial.empty: return partial.sample(1)

    loose = df[
        (df["skin_tone"] == skin_tone) &
        (df["undertone"] == undertone)
    ]
    if not loose.empty: return loose.sample(1)

    # Prevent ValueError if dataframe is completely empty
    return df.sample(1) if not df.empty else pd.DataFrame()


# ---------------- NORMALIZE SHADE ----------------
def normalize_family(family):
    family = family.lower().strip()

    if any(word in family for word in ["golden", "warm", "cool"]):
        return "beige"

    mapping = {
        "beige": "beige", "tan": "tan", "caramel": "caramel", 
        "ivory": "ivory", "honey": "honey", "espresso": "espresso", 
        "nude": "nude", "pink": "pink", "rose": "rose", 
        "red": "red", "berry": "berry", "wine": "wine", 
        "coral": "coral", "peach": "peach", "brown": "brown"
    }

    for key, val in mapping.items():
        if key in family:
            return val

    return family


# ---------------- PRODUCT MATCH ----------------
def get_product(brand_df, brand, product_type, shade_family):
    shade_family = normalize_family(shade_family)
    
    # Base filter for product type to save redundant filtering
    type_df = brand_df[brand_df["product_type"] == product_type]
    
    if type_df.empty:
        return "No product found"

    # exact match
    result = type_df[
        (type_df["brand"] == brand) &
        (type_df["shade_family"] == shade_family)
    ]

    # partial match
    if result.empty and shade_family:
        result = type_df[
            (type_df["brand"] == brand) &
            (type_df["shade_family"].str.contains(shade_family, case=False, na=False))
        ]

    # ignore brand match
    if result.empty and shade_family:
        result = type_df[
            type_df["shade_family"].str.contains(shade_family, case=False, na=False)
        ]

    # Return result or fallback
    if not result.empty:
        return result.sample(1)["shade_name"].values[0]
    
    return type_df.sample(1)["shade_name"].values[0]


# ---------------- MAIN FUNCTION ----------------
# Added DataFrames as arguments so you can pass the pre-loaded data in
def generate_recommendation(skin_tone, undertone, occasion, weather, brand, rec_df, makeup_df, brand_df):
    
    # Normalize inputs
    skin_tone = skin_tone.lower().strip()
    undertone = undertone.lower().strip()
    occasion, weather, brand = normalize_inputs(occasion, weather, brand)

    # ---------------- RECOMMENDATION ----------------
    filtered_rec = smart_filter(rec_df, skin_tone, undertone, occasion, weather)

    if filtered_rec.empty:
        colors = ""
        avoid = ""
    else:
        # 🔥 FIX: RANDOM ROW
        row = filtered_rec.sample(1).iloc[0]
        colors = row.get("colors", "")
        avoid = row.get("avoid_colors", "")

    colors_list = [c.strip() for c in colors.split(",") if c.strip()]
    selected_color = random.choice(colors_list) if colors_list else "black"

    # ---------------- MAKEUP ----------------
    makeup_match = makeup_df[
        (makeup_df["skin_tone"] == skin_tone) &
        (makeup_df["undertone"] == undertone)
    ]

    if makeup_match.empty and not makeup_df.empty:
        makeup_match = makeup_df

    if not makeup_match.empty:
        make = makeup_match.sample(1).iloc[0]
        foundation_family = make.get("foundation", "")
        lipstick_family = make.get("lipstick", "")
        blush_family = make.get("blush", "")
    else:
        foundation_family = lipstick_family = blush_family = ""

    # ---------------- BRAND MATCH ----------------
    foundation_product = get_product(brand_df, brand, "foundation", foundation_family)
    lipstick_product = get_product(brand_df, brand, "lipstick", lipstick_family)
    blush_product = get_product(brand_df, brand, "blush", blush_family)

    # ---------------- TIP ----------------
    tips = [
        "Use light makeup for daytime",
        "Go bold for parties",
        "Hydrate your skin before makeup",
        "Match lipstick with outfit tone",
        "Less is more for office looks"
    ]

    # ---------------- RETURN ----------------
    return {
        "color": selected_color,
        "all_colors": colors,
        "avoid": avoid,
        "foundation": foundation_product,
        "lipstick": lipstick_product,
        "blush": blush_product,
        "tip": random.choice(tips)
    }
    print("FINAL COLORS:", colors)