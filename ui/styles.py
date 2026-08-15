"""
CSS for the Smart Shopping Negotiator dashboard UI.

Pulled out of app.py so styling can be edited independently of the
Streamlit layout / LangGraph orchestration logic.

Design tokens (kept in one place so every component pulls from the
same palette instead of scattering hex codes across components.py):

    Background   : #0b0d12 (page) / #151822 (card) / #1a1e27 (card-alt)
    Border       : #262b36
    Text         : #f5f6f8 (primary) / #9aa1ad (secondary)
    Accent (gold): #f0b94d  -> top pick, primary CTA, "medium" scores
    Success      : #3fb950  -> high scores, confirmed specs
    Danger       : #e74c3c  -> risk flags, low scores
    Info         : #4a9eff  -> informational notes
"""

CHAT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --bg-page: #0b0d12;
        --bg-card: #151822;
        --bg-card-alt: #1a1e27;
        --border-subtle: #262b36;
        --text-primary: #f5f6f8;
        --text-secondary: #9aa1ad;
        --accent-gold: #f0b94d;
        --accent-gold-soft: rgba(240, 185, 77, 0.12);
        --success: #3fb950;
        --success-soft: rgba(63, 185, 80, 0.12);
        --danger: #e74c3c;
        --danger-soft: rgba(231, 76, 60, 0.12);
        --info: #4a9eff;
        --info-soft: rgba(74, 158, 255, 0.12);
        --font-display: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
    }

    /* ---------------- Page-level ---------------- */
    .stApp { background-color: var(--bg-page); }
    .block-container { max-width: 1100px; padding-top: 1.5rem; }

    h1, h2, h3 { font-family: var(--font-display); }

    [data-testid="stChatMessage"] {
        background-color: var(--bg-card);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border: 1px solid var(--border-subtle);
    }

    /* Buttons -> premium chip look, applies to chips, view-deal, reset, apply */
    .stButton > button, .stLinkButton > a {
        background-color: var(--bg-card-alt) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: border-color 0.15s ease, transform 0.05s ease;
    }
    .stButton > button:hover, .stLinkButton > a:hover {
        border-color: var(--accent-gold) !important;
        color: var(--accent-gold) !important;
    }
    .stLinkButton > a { color: #1a1a1a !important; background-color: var(--accent-gold) !important; font-weight: 700 !important; }
    .stLinkButton > a:hover { color: #1a1a1a !important; filter: brightness(1.08); }

    /* ---------------- Section label ---------------- */
    .section-eyebrow {
        font-size: 12px; color: var(--text-secondary); text-transform: uppercase;
        letter-spacing: 0.08em; margin: 18px 0 8px; font-weight: 600;
    }

    /* ---------------- TOP PICK ---------------- */
    .toppick-card {
        background: linear-gradient(180deg, rgba(240,185,77,0.05), transparent 40%), var(--bg-card);
        border: 1.5px solid var(--accent-gold);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 6px;
    }
    .toppick-badge-row { display:flex; align-items:center; gap:8px; margin-bottom: 14px; }
    .toppick-badge {
        background: var(--accent-gold); color:#1a1a1a; font-weight:800; font-size:11px;
        padding: 4px 10px; border-radius: 7px; letter-spacing:0.04em; font-family: var(--font-display);
    }
    .toppick-subtext { color: var(--success); font-size: 12px; font-weight: 500; }

    .toppick-grid { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
    .toppick-imgwrap {
        width: 220px; flex-shrink:0; border-radius: 12px; overflow:hidden; background:#0d0f14;
        border: 1px solid var(--border-subtle); display:flex; align-items:center; justify-content:center;
        min-height: 220px;
    }
    .toppick-imgwrap img { width:100%; height:100%; object-fit:contain; display:block; }
    .toppick-noimg { font-size: 42px; color: #3a3f4a; }

    .toppick-body { flex: 1 1 280px; min-width: 240px; }
    .toppick-title { font-family: var(--font-display); font-weight:700; font-size:17px; color:var(--text-primary); margin:0 0 8px; line-height:1.4; }

    .tag-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom: 10px; }
    .tag-good { background: var(--success-soft); color: var(--success); font-size:11px; font-weight:600; padding:4px 9px; border-radius:6px; }
    .tag-warn { background: var(--accent-gold-soft); color: var(--accent-gold); font-size:11px; font-weight:600; padding:4px 9px; border-radius:6px; }

    .toppick-price { font-family: var(--font-display); font-size:26px; font-weight:800; color:var(--text-primary); margin: 6px 0 0; }
    .toppick-store { font-size:12px; color:var(--text-secondary); margin: 0 0 10px; }
    .toppick-desc { font-size:13px; color:#cfd3da; line-height:1.6; margin:0; }

    .toppick-scorepanel {
        flex: 0 0 200px; background: var(--bg-card-alt); border: 1px solid var(--border-subtle);
        border-radius: 12px; padding: 14px 16px; min-width: 180px;
    }
    .scorepanel-label { font-size:11px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px; }
    .scorepanel-overall { font-family: var(--font-display); font-size:32px; font-weight:800; line-height:1.1; margin-bottom:12px; }
    .scorepanel-overall .suffix { font-size:15px; color:var(--text-secondary); font-weight:600; }

    .score-row { display:flex; align-items:center; justify-content:space-between; padding: 5px 0; border-top: 1px solid var(--border-subtle); }
    .score-row:first-of-type { border-top: none; }
    .score-row-label { font-size:12px; color:#cfd3da; }
    .score-pill { font-size:11px; font-weight:700; padding: 2px 8px; border-radius: 6px; }
    .score-pill-high { background: var(--success-soft); color: var(--success); }
    .score-pill-mid  { background: var(--accent-gold-soft); color: var(--accent-gold); }
    .score-pill-low  { background: var(--danger-soft); color: var(--danger); }

    /* ---------------- Alternatives ---------------- */
    .alt-card { background: var(--bg-card-alt); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 12px; display:flex; flex-direction:column; gap:8px; }
    .alt-card-suspicious { border-color: var(--danger); }
    .alt-imgwrap { width:100%; height:120px; border-radius:8px; overflow:hidden; background:#0d0f14; display:flex; align-items:center; justify-content:center; border:1px solid var(--border-subtle); }
    .alt-imgwrap img { width:100%; height:100%; object-fit:contain; }
    .alt-noimg { font-size:26px; color:#3a3f4a; }
    .alt-top-row { display:flex; justify-content:space-between; align-items:flex-start; gap:6px; }
    .alt-title { font-size:13px; font-weight:600; color:var(--text-primary); margin:0; line-height:1.35; }
    .alt-price { font-size:15px; font-weight:700; color:var(--text-primary); white-space:nowrap; }
    .alt-tradeoff { font-size:12px; color:var(--text-secondary); margin:0; line-height:1.5; }
    .alt-tradeoff-flagged { color: var(--danger); }

    /* ---------------- Info banner / filtered note ---------------- */
    .info-banner { background: var(--info-soft); border-left:3px solid var(--info); color:#cfe4ff; font-size:12.5px; padding:8px 12px; border-radius:6px; margin: 10px 0; }

    /* ---------------- Bottom info row (3 cards) ---------------- */
    .info-card { background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 16px; }
    .info-card-title { font-family: var(--font-display); font-size:12.5px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.04em; margin: 0 0 12px; display:flex; align-items:center; gap:6px; }

    .checklist-item { display:flex; align-items:flex-start; gap:8px; font-size:12.5px; color:#cfd3da; margin-bottom:9px; line-height:1.5; }
    .checklist-item:last-child { margin-bottom:0; }
    .checklist-icon-good { color: var(--success); flex-shrink:0; }
    .checklist-icon-warn { color: var(--accent-gold); flex-shrink:0; }

    .trust-ring-wrap { display:flex; align-items:center; gap:16px; margin-bottom: 12px; }
    .trust-ring { width:76px; height:76px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
    .trust-ring-inner { width:60px; height:60px; border-radius:50%; background:var(--bg-card); display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .trust-ring-value { font-family: var(--font-display); font-size:17px; font-weight:800; color:var(--text-primary); }
    .trust-ring-max { font-size:9px; color:var(--text-secondary); }
    .trust-risk-label { font-size:13px; font-weight:700; font-family: var(--font-display); }

    .keyspec-row { display:flex; align-items:flex-start; gap:8px; font-size:12.5px; color:#cfd3da; margin-bottom:9px; line-height:1.5; }
    .keyspec-row:last-child { margin-bottom:0; }

    /* ---------------- Bottom line ---------------- */
    .bottomline-box { background: var(--bg-card-alt); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 16px; margin: 10px 0; }
    .bottomline-title { font-size:12.5px; font-weight:700; color:var(--accent-gold); text-transform:uppercase; letter-spacing:0.04em; margin: 0 0 6px; display:flex; align-items:center; gap:6px; }
    .bottomline-text { font-size:13.5px; color:var(--text-primary); line-height:1.6; margin:0; }

    /* ---------------- Ask-next section ---------------- */
    .ask-next-title { font-size:12.5px; color:var(--text-secondary); display:flex; align-items:center; gap:6px; margin: 4px 0 8px; }

    /* ---------------- Star rating ---------------- */
    .star-rating { color:#f0b94d; font-size:14px; letter-spacing:1px; }
    .star-rating-count { color:var(--text-secondary); font-size:12px; margin-left:6px; }

    /* ---------------- Expander (used for score formula) ---------------- */
    [data-testid="stExpander"] {
        background: var(--bg-card-alt);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        margin-bottom: 14px;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--accent-gold) !important; }

    @media (max-width: 640px) {
        .toppick-imgwrap { width: 100%; }
        .toppick-scorepanel { flex-basis: 100%; }
    }
</style>
"""


