CSS = """
<style>
:root{--brand:#1f4e9c;--brand2:#3b82f6;--ok:#15803d;--warn:#b45309;--bad:#b91c1c;--card:rgba(120,130,150,.10);--line:rgba(120,130,150,.28);}
.block-container{padding-top:1.4rem;max-width:1180px;}
.hero{background:linear-gradient(120deg,#12306b 0%,#1f4e9c 55%,#3b82f6 100%);color:#fff;border-radius:16px;padding:20px 26px;margin-bottom:14px;}
.hero h1{margin:0;font-size:1.55rem;font-weight:700;color:#fff;}
.hero p{margin:4px 0 0;opacity:.9;font-size:.95rem;color:#fff;}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.74rem;font-weight:600;margin:0 6px 4px 0;border:1px solid var(--line);background:var(--card);}
.pill.ok{color:var(--ok);border-color:var(--ok);} .pill.bad{color:var(--bad);border-color:var(--bad);} .pill.warn{color:var(--warn);border-color:var(--warn);}
.chip{display:inline-block;padding:2px 9px;border-radius:8px;font-size:.78rem;margin:2px 6px 2px 0;background:rgba(59,130,246,.14);border:1px solid rgba(59,130,246,.35);}
.chip.dim{background:var(--card);border-color:var(--line);}
.src-meta{font-size:.82rem;opacity:.85;margin-bottom:6px;}
.evidence{white-space:pre-wrap;font-size:.86rem;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;max-height:340px;overflow:auto;}
.trace-row{display:flex;gap:10px;align-items:baseline;font-size:.86rem;padding:3px 0;border-bottom:1px dashed var(--line);}
.trace-row .n{min-width:150px;font-weight:600;} .trace-row .ms{opacity:.6;min-width:60px;text-align:right;}
.disclaimer{font-size:.8rem;opacity:.8;border-left:3px solid var(--brand2);padding:4px 10px;margin-top:8px;}
.stChatMessage{border-radius:14px;}
section[data-testid="stSidebar"] .stButton>button{width:100%;text-align:left;}
</style>
"""
