"""
BotTrainer — Custom CSS Styles.
=================================
All Streamlit CSS overrides for the dark-themed production UI.
Extracted from app.py for maintainability.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111119;
    --bg-card: rgba(17,17,25,0.85);
    --bg-card-hover: rgba(25,25,40,0.95);
    --border-subtle: rgba(255,255,255,0.06);
    --border-hover: rgba(102,126,234,0.35);
    --accent-primary: #667eea;
    --accent-secondary: #764ba2;
    --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent-gradient-h: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    --text-primary: #e8eaed;
    --text-secondary: #9aa0b0;
    --text-muted: #6b7280;
    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-full: 9999px;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 30px rgba(102,126,234,0.15);
    --transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif \!important;
}
.main { background: var(--bg-primary) !important; }
.main .block-container { padding: 2.5rem 3rem 4rem 3rem; max-width: 1280px; }
.stApp { background: var(--bg-primary); }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(102,126,234,0.45); }

#MainMenu { display: none !important; }
footer { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0b1d 0%, #161432 40%, #1a1640 100%) !important;
    border-right: 1px solid rgba(102,126,234,0.1) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    padding: 11px 16px !important; border-radius: var(--radius-md) !important;
    transition: var(--transition) !important; font-weight: 500 !important;
    font-size: 0.92rem !important; border: 1px solid transparent !important;
    cursor: pointer !important; margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(102,126,234,0.12) !important; border-color: rgba(102,126,234,0.2) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(102,126,234,0.15) !important; border-color: rgba(102,126,234,0.35) !important;
    font-weight: 600 !important; box-shadow: 0 0 20px rgba(102,126,234,0.1);
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 1.2px;
    font-weight: 600 !important; color: var(--text-muted) !important;
}

.stTextInput > div > div {
    background: var(--bg-secondary) !important; border: 1.5px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important; transition: var(--transition) !important;
    box-shadow: var(--shadow-sm);
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15), var(--shadow-md) !important;
}
.stTextInput input {
    color: var(--text-primary) !important; font-size: 1rem !important;
    padding: 14px 18px !important; font-weight: 400 !important;
}
.stTextInput input::placeholder { color: var(--text-muted) !important; font-weight: 400 !important; }
.stTextInput label { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.85rem !important; }

.stSelectbox > div > div {
    background: var(--bg-secondary) !important; border: 1.5px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important; transition: var(--transition) !important;
}
.stSelectbox > div > div:hover { border-color: rgba(102,126,234,0.3) !important; }
.stSelectbox [data-baseweb="select"] span { color: var(--text-primary) !important; }
.stSelectbox label { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
[data-baseweb="popover"] {
    background: var(--bg-secondary) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important; box-shadow: var(--shadow-lg) !important;
}
[data-baseweb="popover"] li { color: var(--text-primary) !important; transition: var(--transition); }
[data-baseweb="popover"] li:hover { background: rgba(102,126,234,0.12) !important; }
[data-baseweb="popover"] li[aria-selected="true"] { background: rgba(102,126,234,0.2) !important; }

.stSlider label { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-primary) !important; border-color: var(--accent-primary) !important;
    box-shadow: 0 0 10px rgba(102,126,234,0.4) !important; width: 20px !important; height: 20px !important;
}
.stSlider [data-baseweb="slider"] > div > div:first-child { background: rgba(102,126,234,0.15) !important; }
.stSlider [data-baseweb="slider"] > div > div > div:first-child { background: var(--accent-gradient) !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important; font-size: 0.8rem !important;
}
.stSlider [data-baseweb="slider"] [data-testid="StyledThumbValue"] {
    color: var(--text-primary) !important; font-weight: 600 !important; font-size: 0.85rem !important;
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Inter', sans-serif \!important; font-weight: 600 \!important;
    font-size: 0.92rem !important; border-radius: var(--radius-md) !important;
    padding: 10px 24px !important; transition: var(--transition) !important;
    letter-spacing: 0.2px; border: 1.5px solid var(--border-subtle) !important;
    background: var(--bg-secondary) !important; color: var(--text-primary) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent-primary) !important; background: rgba(102,126,234,0.08) !important;
    box-shadow: 0 4px 16px rgba(102,126,234,0.15) !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: var(--accent-gradient) !important; border: none !important;
    border-radius: var(--radius-md) !important; padding: 12px 32px !important;
    font-weight: 700 !important; font-size: 0.95rem !important; color: #fff !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(102,126,234,0.5) !important; transform: translateY(-2px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px); box-shadow: 0 2px 12px rgba(102,126,234,0.3) !important;
}
.stDownloadButton > button {
    background: rgba(52,211,153,0.08) !important; border-color: rgba(52,211,153,0.25) !important;
    color: var(--success) !important;
}
.stDownloadButton > button:hover {
    background: rgba(52,211,153,0.15) !important; border-color: var(--success) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg); padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important; padding: 10px 22px !important;
    font-weight: 500 !important; font-size: 0.88rem !important;
    color: var(--text-secondary) !important; transition: var(--transition) !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(102,126,234,0.08) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] {
    background: var(--accent-gradient) !important; color: #fff !important;
    font-weight: 600 !important; box-shadow: 0 2px 12px rgba(102,126,234,0.3);
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.streamlit-expanderHeader {
    font-weight: 600 !important; font-size: 0.92rem !important;
    color: var(--text-primary) !important; background: rgba(255,255,255,0.02) !important;
    border-radius: var(--radius-md) !important; border: 1px solid var(--border-subtle) !important;
    transition: var(--transition) !important; padding: 14px 18px !important;
}
.streamlit-expanderHeader:hover { background: rgba(102,126,234,0.06) !important; border-color: rgba(102,126,234,0.2) !important; }
details[open] > .streamlit-expanderHeader {
    border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important;
    border-color: rgba(102,126,234,0.25) !important; background: rgba(102,126,234,0.06) !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.01) !important; border: 1px solid var(--border-subtle) !important;
    border-top: none !important; border-bottom-left-radius: var(--radius-md) !important;
    border-bottom-right-radius: var(--radius-md) !important; padding: 16px 18px !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-md) !important; overflow: hidden; }
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] { border-radius: var(--radius-md) !important; }

.stProgress > div > div > div > div {
    background: var(--accent-gradient) !important; border-radius: var(--radius-full) !important;
    box-shadow: 0 0 12px rgba(102,126,234,0.3);
}
.stProgress > div > div > div { background: rgba(255,255,255,0.04) !important; border-radius: var(--radius-full) !important; }
.stProgress p { color: var(--text-secondary) !important; font-size: 0.82rem !important; }

.stAlert { border-radius: var(--radius-md) !important; border: none !important; font-size: 0.9rem !important; }
[data-testid="stAlert"][data-baseweb*="notification"] { border-radius: var(--radius-md) !important; }
div[data-testid="stNotification"] { border-radius: var(--radius-md) !important; backdrop-filter: blur(12px); }

.stSpinner > div { border-top-color: var(--accent-primary) !important; }
.stSpinner + div { color: var(--text-secondary) !important; }

[data-testid="stJson"] {
    background: var(--bg-secondary) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important; padding: 16px !important;
}

[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.01); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg); padding: 8px;
}

.main .stMarkdown p, .main .stMarkdown li { color: var(--text-secondary); }
.main .stMarkdown strong { color: var(--text-primary); }
.main .stMarkdown code {
    background: rgba(102,126,234,0.1); color: var(--accent-primary);
    padding: 2px 7px; border-radius: 5px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em;
}

.hero-section { margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid var(--border-subtle); }
.hero-title {
    font-size: 2.6rem; font-weight: 900; background: var(--accent-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 6px; letter-spacing: -0.8px; line-height: 1.1;
}
.hero-subtitle { font-size: 1rem; color: var(--text-muted); margin-bottom: 0; font-weight: 400; line-height: 1.55; max-width: 700px; }

.glass-card {
    background: var(--bg-card); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border-subtle); border-radius: var(--radius-lg); padding: 28px;
    margin-bottom: 16px; transition: var(--transition); position: relative; overflow: hidden;
}
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-gradient-h); opacity: 0; transition: opacity 0.3s ease;
}
.glass-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-glow); background: var(--bg-card-hover); }
.glass-card:hover::before { opacity: 1; }

.card-label {
    font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 1.8px; font-weight: 700; margin-bottom: 14px;
    display: flex; align-items: center; gap: 6px;
}
.card-label::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent-primary); box-shadow: 0 0 8px rgba(102,126,234,0.5);
}
