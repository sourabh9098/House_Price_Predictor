import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Background + Global CSS ───────────────────────────────────────────────────
BG_URL = "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=1800&q=80"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── background ── */
.stApp {{
    background: url('{BG_URL}') center center / cover no-repeat fixed;
    font-family: 'DM Sans', sans-serif;
}}
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(8, 14, 26, 0.82);
    z-index: 0;
}}
.block-container {{
    position: relative;
    z-index: 1;
    max-width: 780px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

/* ── hide default chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── hero ── */
.hero {{
    text-align: center;
    padding: 2.6rem 1.5rem 2rem;
    margin-bottom: 1.8rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    backdrop-filter: blur(14px);
}}
.hero-icon {{ font-size: 3rem; margin-bottom: 0.4rem; }}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f5e6c8 30%, #e8a550 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}}
.hero-sub {{
    color: rgba(210,195,175,0.75);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}}

/* ── glass card wrapper ── */
.glass-card {{
    background: rgba(15, 22, 40, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 1.6rem 1.8rem 1.2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
}}

/* ── section label ── */
.section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e8a550;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(232,165,80,0.25);
}}

/* ── widget label overrides ── */
label[data-testid="stWidgetLabel"] p {{
    color: #c8bfad !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* selectbox styling */
div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #e0d8cc !important;
}}

/* ── predict button ── */
div[data-testid="stButton"] button {{
    width: 100%;
    background: linear-gradient(135deg, #c47f17, #e8a550, #c47f17);
    background-size: 200% auto;
    color: #1a1008;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    border: none;
    border-radius: 50px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.35s ease;
    box-shadow: 0 4px 24px rgba(232,165,80,0.35);
    margin-top: 0.5rem;
    text-transform: uppercase;
}}
div[data-testid="stButton"] button:hover {{
    background-position: right center;
    box-shadow: 0 8px 36px rgba(232,165,80,0.55);
    transform: translateY(-2px);
}}

/* ── result card ── */
.result-card {{
    background: linear-gradient(135deg, rgba(20,30,50,0.92), rgba(30,15,10,0.92));
    border: 1px solid rgba(232,165,80,0.45);
    border-radius: 22px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    margin-top: 1.8rem;
    backdrop-filter: blur(20px);
    animation: popIn 0.55s cubic-bezier(0.34,1.56,0.64,1) both;
}}
@keyframes popIn {{
    from {{ opacity:0; transform: scale(0.88) translateY(20px); }}
    to   {{ opacity:1; transform: scale(1)    translateY(0);    }}
}}
.result-eyebrow {{
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: rgba(210,185,140,0.6);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
.result-price {{
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f7e8c0, #e8a550, #f7e8c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}}
.result-tag {{
    display: inline-block;
    background: rgba(232,165,80,0.14);
    border: 1px solid rgba(232,165,80,0.35);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 0.3rem 1.1rem;
    border-radius: 50px;
    margin-top: 0.9rem;
    text-transform: uppercase;
}}

/* ── insight metrics ── */
.metrics-row {{
    display: flex;
    gap: 0.9rem;
    margin-top: 1.2rem;
}}
.metric-box {{
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 0.85rem 0.6rem;
    text-align: center;
}}
.m-val {{
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8c87a;
}}
.m-lbl {{
    font-size: 0.65rem;
    color: rgba(200,185,155,0.55);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}}

/* ── footer ── */
.footer {{
    text-align: center;
    color: rgba(180,165,140,0.35);
    font-size: 0.72rem;
    margin-top: 2.5rem;
    letter-spacing: 0.5px;
}}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m = joblib.load("ridge_model.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

try:
    model, scaler = load_artifacts()
    ok = True
except Exception as e:
    ok = False
    err = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🏡</div>
  <div class="hero-title">House Price Predictor</div>
  <div class="hero-sub">Enter your property details below and get an instant AI-powered valuation</div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"⚠️ Could not load model files. Place `ridge_model.pkl` & `scaler.pkl` in the same directory.\n\n{err}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Section 1: Core Property ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🏠 Core Property Details</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6,
                             help="Overall material and finish quality of the house")
    gr_liv_area  = st.slider("Above-Grade Living Area (sq ft)", 400, 5000, 1500, 10,
                             help="Total living area above ground level")
with c2:
    full_bath    = st.slider("Full Bathrooms", 0, 4, 2)
    bedroom      = st.slider("Bedrooms Above Grade", 0, 8, 3)
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Garage & Basement ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🚗 Garage & Basement</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    garage_cars   = st.slider("Garage Capacity (cars)", 0, 4, 2)
    total_bsmt_sf = st.slider("Total Basement Area (sq ft)", 0, 3000, 1000, 10)
with c4:
    first_flr_sf  = st.slider("1st Floor Area (sq ft)", 300, 4000, 1100, 10)
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Style & Roof ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🏗️ Style & Roof</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    house_style = st.selectbox("House Style", [
        "1Story", "2Story", "1.5Unf", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"
    ])
with c6:
    roof_style  = st.selectbox("Roof Style", [
        "Gable", "Hip", "Gambrel", "Mansard", "Shed"
    ])
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 4: Sale & Neighbourhood ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📍 Sale & Neighbourhood</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    sale_condition = st.selectbox("Sale Condition", [
        "Normal", "Partial", "Family", "Alloca", "AdjLand"
    ])
with c8:
    neighborhood = st.selectbox("Neighborhood", [
        "CollgCr","OldTown","Edwards","Somerst","NridgHt","Gilbert","Sawyer",
        "NWAmes","SawyerW","Mitchel","BrkSide","Crawfor","IDOTRR","NAmes",
        "Timber","NoRidge","Veenker","ClearCr","NPkVill","Blueste","Greens",
        "BrDale","SWISU","MeadowV","GrnHill","Landmrk","StoneBr"
    ])
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_features():
    vec = {
        'Overall Qual':  overall_qual,
        'Gr Liv Area':   gr_liv_area,
        'Garage Cars':   garage_cars,
        'Total Bsmt SF': total_bsmt_sf,
        '1st Flr SF':    first_flr_sf,
        'Full Bath':     full_bath,
        'Bedroom AbvGr': bedroom,
    }
    for cat in ['AdjLand','Alloca','Family','Normal','Partial']:
        vec[f'Sale Condition_{cat}'] = 1 if sale_condition == cat else 0
    for sty in ['1.5Unf','1Story','2.5Fin','2.5Unf','2Story','SFoyer','SLvl']:
        vec[f'House Style_{sty}'] = 1 if house_style == sty else 0
    for nb in [
        'Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards',
        'Gilbert','Greens','GrnHill','IDOTRR','Landmrk','MeadowV','Mitchel',
        'NAmes','NPkVill','NWAmes','NoRidge','NridgHt','OldTown','SWISU',
        'Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker'
    ]:
        vec[f'Neighborhood_{nb}'] = 1 if neighborhood == nb else 0
    for rs in ['Gable','Gambrel','Hip','Mansard','Shed']:
        vec[f'Roof Style_{rs}'] = 1 if roof_style == rs else 0

    ordered = scaler.feature_names_in_
    return np.array([[vec.get(f, 0) for f in ordered]])

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔍  ESTIMATE MY PROPERTY VALUE"):
    with st.spinner("Calculating valuation…"):
        try:
            X     = build_features()
            X_sc  = scaler.transform(X)
            price = float(model.predict(X_sc)[0])
            price = max(price, 0)

            if price < 100_000:
                seg, seg_col = "Starter Home",  "#6ee7b7"
            elif price < 200_000:
                seg, seg_col = "Mid-Range",     "#60a5fa"
            elif price < 350_000:
                seg, seg_col = "Premium",       "#e8a550"
            else:
                seg, seg_col = "Luxury Estate", "#f87171"

            st.markdown(f"""
            <div class="result-card">
              <div class="result-eyebrow">Estimated Property Value</div>
              <div class="result-price">${price:,.0f}</div>
              <div class="result-tag" style="color:{seg_col};border-color:{seg_col}55;">
                {seg}
              </div>
            </div>
            """, unsafe_allow_html=True)

            price_per_sqft = price / gr_liv_area if gr_liv_area else 0
            rooms          = bedroom + full_bath
            price_per_room = price / rooms if rooms else 0
            qual_ratio     = round(price / (overall_qual * 10_000), 2)

            st.markdown(f"""
            <div class="metrics-row">
              <div class="metric-box">
                <div class="m-val">${price_per_sqft:,.0f}</div>
                <div class="m-lbl">Per Sq Ft</div>
              </div>
              <div class="metric-box">
                <div class="m-val">${price_per_room:,.0f}</div>
                <div class="m-lbl">Per Room</div>
              </div>
              <div class="metric-box">
                <div class="m-val">{qual_ratio}x</div>
                <div class="m-lbl">Quality Ratio</div>
              </div>
              <div class="metric-box">
                <div class="m-val">{overall_qual}/10</div>
                <div class="m-lbl">Quality Score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  House Price Predictor &nbsp;·&nbsp; Ridge Regression Model &nbsp;·&nbsp; Built with Streamlit<br>
  Trained on Ames Housing Dataset · 51 engineered features
</div>
""", unsafe_allow_html=True)

