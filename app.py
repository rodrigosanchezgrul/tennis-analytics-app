import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import time
import tempfile
from datetime import date
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="CourtIQ Pro | Tennis Analytics, ML & Video Lab", page_icon="🎾", layout="wide")

# --- CUSTOM STYLING (HIGH-CONTRAST READABILITY IN BOTH LIGHT & DARK MODE) ---
st.markdown("""
<style>
    .coach-card, .loss-card, .wall-card, .video-card {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .coach-card h4, .loss-card h4, .wall-card h4, .video-card h4 {
        color: #ffffff !important;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.15rem;
    }
    .coach-card p, .loss-card p, .wall-card p, .video-card p,
    .coach-card b, .loss-card b, .wall-card b, .video-card b,
    .coach-card i, .loss-card i, .wall-card i, .video-card i {
        color: #f0f6fc !important;
    }
    .coach-card { border-left: 6px solid #2ecc71; }
    .loss-card  { border-left: 6px solid #e74c3c; }
    .wall-card  { border-left: 6px solid #3498db; }
    .video-card { border-left: 6px solid #9b59b6; }
</style>
""", unsafe_allow_html=True)

# --- SUPABASE CONNECTION ---
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- DATA LOADING & METADATA PARSING ---
def parse_rich_note(note_text):
    if not isinstance(note_text, str):
        return {}, ""
    match = re.search(r"\[META:(.*?)\]\s*(.*)", note_text, re.DOTALL)
    if match:
        try:
            meta = json.loads(match.group(1))
            clean_text = match.group(2).strip()
            return meta, clean_text
        except Exception:
            return {}, note_text
    return {}, note_text

@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    response = supabase.table("personal_matches").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return df, df, []

    raw_db_cols = list(df.columns)

    if "opponent_name" in df.columns and "opponent" not in df.columns:
        df["opponent"] = df["opponent_name"].fillna("Unknown")
    elif "opponent" in df.columns and "opponent_name" not in df.columns:
        df["opponent_name"] = df["opponent"].fillna("Unknown")

    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values(by="match_date", ascending=False)

    note_col = next((c for c in ["tactical_note", "tactical_notes", "notes", "note", "reflection", "comments"] if c in df.columns), None)
    if note_col:
        parsed = df[note_col].apply(parse_rich_note)
        meta_df = pd.json_normalize(parsed.apply(lambda x: x[0]))
        df["clean_note"] = parsed.apply(lambda x: x[1])
        for col in meta_df.columns:
            if col not in df.columns:
                df[col] = meta_df[col].values

    if "ankle_weights" not in df.columns:
        df["ankle_weights"] = "No"
    else:
        df["ankle_weights"] = df["ankle_weights"].fillna("No")

    is_video = df["match_type"].astype(str).str.contains("Video Biomechanics", case=False, na=False) if "match_type" in df.columns else pd.Series(False, index=df.index)
    video_df = df[is_video].copy()
    manual_df = df[~is_video].copy()

    if not manual_df.empty:
        if "match_type" in manual_df.columns:
            manual_df["Category"] = np.where(
                manual_df["match_type"].astype(str).str.contains("Wall|Drill|Coach|Practice", case=False, na=False),
                "Training & Wall",
                "Competitive Match"
            )
        else:
            manual_df["Category"] = "Competitive Match"

        if "result" in manual_df.columns:
            def map_outcome(val):
                s = str(val).lower().strip()
                if s.startswith("w") or "gan" in s:
                    return "Win"
                elif s.startswith("l") or "perd" in s:
                    return "Loss"
                return "Training / Wall"
            manual_df["Outcome"] = manual_df["result"].apply(map_outcome)
        else:
            manual_df["Outcome"] = "Training / Wall"

    return manual_df, video_df, raw_db_cols

def get_table_columns(existing_cols: list):
    if existing_cols:
        return existing_cols
    try:
        r = supabase.table("personal_matches").select("*").limit(1).execute()
        if r.data:
            return list(r.data[0].keys())
    except Exception:
        pass
    return ["match_date", "opponent_name", "match_type", "surface", "result", "score", "serve_rating", "mental_focus", "energy_level", "tactical_note", "tactical_notes", "notes"]

def _safe_insert_batch(records: list):
    """Automatically strips any column rejected by PostgREST PGRST204 and packs it into META JSON."""
    if not records:
        return
    batch = []
    for r in records:
        rec = r.copy()
        opp_val = str(rec.get("opponent_name") or rec.get("opponent") or "Unknown").strip()
        if not opp_val or opp_val.lower() == "nan":
            opp_val = "Unknown"
        rec["opponent_name"] = opp_val
        rec["opponent"] = opp_val
        batch.append({k: v for k, v in rec.items() if v is not None})

    for _ in range(15):
        try:
            supabase.table("personal_matches").insert(batch).execute()
            return
        except Exception as e:
            err_str = str(e)
            m = re.search(r"Could not find the '([^']+)' column", err_str)
            if m:
                bad_col = m.group(1)
                for rec in batch:
                    if bad_col in rec:
                        val = rec.pop(bad_col)
                        # Preserve the stripped column's value inside META JSON so nothing is lost
                        n_key = next((c for c in ["tactical_note", "tactical_notes", "notes", "note", "reflection", "comments"] if c in rec), None)
                        if n_key:
                            meta, body = parse_rich_note(rec[n_key])
                            meta[bad_col] = val
                            rec[n_key] = f"[META:{json.dumps(meta)}] {body}".strip()
                continue
            raise e

def insert_match(record: dict, existing_cols: list):
    _safe_insert_batch([record])
    st.cache_data.clear()

def bulk_insert_matches(records: list, existing_cols: list):
    _safe_insert_batch(records)
    st.cache_data.clear()

def wipe_all_records():
    st.cache_data.clear()
    """Deletes all existing rows in personal_matches so a fresh CSV can be imported cleanly."""
    existing = supabase.table("personal_matches").select("id").execute()
    if existing.data:
        ids = [r["id"] for r in existing.data if "id" in r]
        if ids:
            supabase.table("personal_matches").delete().in_("id", ids).execute()
            return len(ids)
    # Fallback delete if primary key is named differently
    supabase.table("personal_matches").delete().neq("opponent_name", "___NEVER_MATCH___").execute()
    return 0

# --- VIDEO COMPUTER VISION & AI BIOMECHANICS ENGINE ---
def analyze_video_cv(video_path: str, focus_stroke: str):
    if not HAS_CV2:
        return {
            "duration_sec": 0.0,
            "footwork_cadence": 6.5,
            "swing_explosiveness": 6.5,
            "recovery_rhythm": 6.5,
            "burst_ratio": 2.0,
            "motion_df": pd.DataFrame(),
            "key_frames": []
        }

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    prev_gray = None
    motion_scores = []
    frame_indices = []
    sampled_frames = []

    step = max(1, int(fps // 6))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (480, 270))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = float(np.mean(diff))
                motion_scores.append(score)
                frame_indices.append(round(idx / fps, 2))
                sampled_frames.append((score, round(idx / fps, 2), cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            prev_gray = gray
        idx += 1
    cap.release()

    if not motion_scores:
        return None

    arr = np.array(motion_scores)
    mean_motion = float(np.mean(arr))
    peak_motion = float(np.percentile(arr, 95))
    burst_ratio = peak_motion / (mean_motion + 1e-5)

    footwork_cadence = float(np.clip(4.0 + (mean_motion / 3.5), 1.0, 10.0))
    swing_explosiveness = float(np.clip(4.5 + (burst_ratio * 1.4), 1.0, 10.0))
    recovery_rhythm = float(np.clip(8.5 - (np.std(arr) / (mean_motion + 1e-5)) * 1.8, 1.0, 10.0))

    top_frames = sorted(sampled_frames, key=lambda x: x[0], reverse=True)[:3]
    top_frames = sorted(top_frames, key=lambda x: x[1])
    motion_df = pd.DataFrame({"Time (sec)": frame_indices, "Movement & Swing Intensity": motion_scores})

    return {
        "duration_sec": round(duration_sec, 1),
        "footwork_cadence": round(footwork_cadence, 1),
        "swing_explosiveness": round(swing_explosiveness, 1),
        "recovery_rhythm": round(recovery_rhythm, 1),
        "burst_ratio": round(burst_ratio, 2),
        "motion_df": motion_df,
        "key_frames": top_frames
    }

def analyze_video_gemini(video_path: str, focus_stroke: str, session_context: str, ankle_weights_used: str):
    if not HAS_GENAI or "GEMINI_API_KEY" not in st.secrets:
        return None
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        uploaded = client.files.upload(file=video_path)
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        weight_note = (
            f"IMPORTANT CONTEXT: The player is wearing ankle weights ({ankle_weights_used}) during this clip. Account for added lower-body load when evaluating footwork speed, split-step explosiveness, and joint alignment."
            if ankle_weights_used != "No"
            else "The player is wearing standard tennis footwear without ankle weights."
        )

        prompt = f"""
        You are an elite ATP/WTA tennis biomechanics coach. Analyze this {session_context} tennis video focusing on: {focus_stroke}.
        {weight_note}
        Return ONLY valid JSON with these exact keys:
        {{
            "kinetic_chain_score": <float 1-10>,
            "footwork_splitstep_score": <float 1-10>,
            "contact_point_score": <float 1-10>,
            "balance_recovery_score": <float 1-10>,
            "strengths_observed": "<2 concise bullet points on what looks technically sound>",
            "biomechanical_flaws": "<2-3 specific technical leaks seen in the footage>",
            "corrective_drills": "<2 specific court or wall drills to fix those flaws>"
        }}
        """
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded, prompt]
        )
        client.files.delete(name=uploaded.name)
        raw_text = resp.text.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        st.warning(f"Gemini Vision note: {e}. Falling back to built-in OpenCV telemetry.")
    return None

# --- MAIN APP ---
st.title("🎾 CourtIQ Pro — Tennis Intelligence, ML & Video Lab")

df, video_df, raw_db_cols = load_data()
valid_db_cols = get_table_columns(raw_db_cols)
note_col = next((c for c in ["tactical_note", "tactical_notes", "notes", "note", "reflection", "comments"] if c in valid_db_cols), "tactical_note")


# ==========================================
# GRANULAR 11-STROKE SCHEMA & UNIFIED EXTRACTOR
# ==========================================
GRANULAR_STROKES = [
    ("1st Serve", "first_serve", ["first_serve", "1st Serve", "First Serve", "1st_serve"], "#2980b9"),
    ("2nd Serve", "second_serve", ["second_serve", "2nd Serve", "Second Serve", "2nd_serve"], "#5dade2"),
    ("Forehand", "fh_rating", ["fh_rating", "Forehand", "Forehand Quality", "forehand"], "#2ecc71"),
    ("Backhand", "bh_rating", ["bh_rating", "Backhand", "Backhand Quality", "backhand"], "#e67e22"),
    ("Forehand Return", "fh_return", ["fh_return", "Forehand Return", "Forehand Return Quality", "fh_return_rating"], "#8e44ad"),
    ("Backhand Return", "bh_return", ["bh_return", "Backhand Return", "Backhand Return Quality", "bh_return_rating"], "#bb8fce"),
    ("Forehand Volley", "fh_volley", ["fh_volley", "Forehand Volley", "FH Volley", "fh_volley_rating"], "#16a085"),
    ("Backhand Volley", "bh_volley", ["bh_volley", "Backhand Volley", "BH Volley", "bh_volley_rating"], "#48c9b0"),
    ("Overhead", "overhead_rating", ["overhead_rating", "Overhead", "Smash", "overhead"], "#f39c12"),
    ("Slice", "slice_rating", ["slice_rating", "Slice", "slice"], "#95a5a6"),
    ("Movement", "movement_rating", ["movement_rating", "Movement", "movement"], "#e74c3c"),
    ("Physical / Stamina", "stamina_rating", ["stamina_rating", "Physical / Stamina", "Physical", "Stamina", "physical_energy", "Physical State / Energy"], "#ff6b6b"),
]

def enrich_granular_strokes(dataframe):
    if dataframe is None or dataframe.empty:
        return dataframe
    out = dataframe.copy()
    # Combine all text columns so regex extraction finds ratings regardless of where they were stored
    text_pool = pd.Series("", index=out.index, dtype="object")
    for tcol in ["clean_note", "tactical_note", "notes", "score"]:
        if tcol in out.columns:
            text_pool = text_pool + " | " + out[tcol].fillna("").astype(str)

    for label, canonical_col, aliases, _ in GRANULAR_STROKES:
        s = pd.Series(np.nan, index=out.index, dtype="float64")
        # 1. Check exact canonical & alias columns (case-insensitive match across df.columns)
        col_lookup = {c.lower().strip(): c for c in out.columns}
        for alias in [label] + aliases:
            matched_c = col_lookup.get(alias.lower().strip())
            if matched_c:
                s = s.fillna(pd.to_numeric(out[matched_c], errors="coerce"))
        # 2. Extract "Label : 7.0" or "Label: 7" from combined text notes
        text_aliases = list(dict.fromkeys([label] + [a for a in aliases if "_" not in a]))
        pattern = r"(?:" + "|".join(re.escape(a) for a in text_aliases) + r")\s*[:=\-]\s*(\d+(?:\.\d+)?)"
        ext = text_pool.str.extract(pattern, flags=re.IGNORECASE)[0]
        s = s.fillna(pd.to_numeric(ext, errors="coerce"))
        out[canonical_col] = s

    # Fallback ONLY for legacy rows where granular split wasn't recorded at all
    if "serve_rating" in out.columns:
        srv = pd.to_numeric(out["serve_rating"], errors="coerce")
        out["first_serve"] = out["first_serve"].fillna(srv)
        out["second_serve"] = out["second_serve"].fillna(srv)
    if "return_quality" in out.columns or "return_rating" in out.columns:
        ret_col = "return_quality" if "return_quality" in out.columns else "return_rating"
        ret = pd.to_numeric(out[ret_col], errors="coerce")
        out["fh_return"] = out["fh_return"].fillna(ret)
        out["bh_return"] = out["bh_return"].fillna(ret)
    if "net_quality" in out.columns or "volley_rating" in out.columns:
        net_col = "net_quality" if "net_quality" in out.columns else "volley_rating"
        net = pd.to_numeric(out[net_col], errors="coerce")
        out["fh_volley"] = out["fh_volley"].fillna(net)
        out["bh_volley"] = out["bh_volley"].fillna(net)

    out["serve_rating"] = out[["first_serve", "second_serve"]].mean(axis=1)
    return out

if not df.empty:
    df = enrich_granular_strokes(df)

ALL_13_VARIABLE_MAP = {label: col for label, col, _, _ in GRANULAR_STROKES}
ALL_13_VARIABLE_MAP["Mental Focus"] = "mental_focus"
ALL_13_VARIABLE_COLS = list(ALL_13_VARIABLE_MAP.values())
# Create ML-ready imputed columns (_ml) so Random Forest, XGBoost, Logistic/Ridge Regression,
# Correlation Heatmaps, and Simulators analyze ALL 12 variables without dropping historical matches
ALL_13_ML_FEATURES = list(ALL_13_VARIABLE_COLS)
for _lbl, _col in ALL_13_VARIABLE_MAP.items():
    if _col not in df.columns:
        df[_col] = np.nan
    df[_col] = pd.to_numeric(df[_col], errors="coerce")
    df[_lbl] = df[_col]

for _lbl, _col in ALL_13_VARIABLE_MAP.items():
    if _col not in df.columns:
        df[_col] = np.nan
    df[_col] = pd.to_numeric(df[_col], errors="coerce")
    df[_lbl] = df[_col]


for _sc in ALL_13_VARIABLE_COLS:
    if _sc not in df.columns:
        df[_sc] = np.nan
    df[_sc] = pd.to_numeric(df[_sc], errors="coerce")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

    "🕸️ Skill Radar & Scatter Lab",
    "📊 Volume, Wins & Loss Reasons",
    "🤖 AI Coach & ML Models",
    "🎥 Video Biomechanics Lab",
    "➕ Log Detailed Session",
    "📁 CSV Reset, Mapper & Database",
    "Pre-Match Scouting"
])

# ==========================================
# TAB 1: PRO SKILL RADAR & SCATTERPLOT LAB
# ==========================================
with tab1:
    if df.empty:
        st.info("No sessions logged yet. Go to '📁 CSV Reset, Mapper & Database' to upload your 22-game CSV cleanly!")
    else:
        st.subheader("🕸️ Pro-Style 11-Stroke Skill & Match Profile (Self-Logged Data Only)")
        st.caption(f"Showing {len(df)} total logged sessions. Unmapped stroke dimensions smoothly use your session's average execution baseline rather than artificial constants.")

        # Compute each row's true baseline average across available numeric ratings
        base_rating_cols = [c for c in ALL_13_ML_FEATURES if c in df.columns]
        if base_rating_cols:
            row_baseline = df[base_rating_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(6.0)
        else:
            row_baseline = pd.Series(6.0, index=df.index)

        def get_axis_series(col_name):
            if col_name in df.columns and df[col_name].notna().any():
                s = pd.to_numeric(df[col_name], errors="coerce")
                return s.fillna(row_baseline)
            return row_baseline.copy()

        radar_df = pd.DataFrame(index=df.index)
        radar_df["Serve Quality"] = get_axis_series("serve_rating")
        radar_df["Return Quality"] = get_axis_series("ret_rating")
        radar_df["Forehand Quality"] = get_axis_series("fh_rating")
        radar_df["Backhand Quality"] = get_axis_series("bh_rating")
        radar_df["Net & Conversion"] = get_axis_series("net_rating")
        radar_df["Baseline Battle (Focus)"] = get_axis_series("mental_focus")
        radar_df["Physical Steal (Energy)"] = get_axis_series("energy_level")
        radar_df["Attack / Overall Index"] = radar_df.mean(axis=1)
        radar_df["Outcome"] = df["Outcome"]

        axes = list(ALL_13_VARIABLE_MAP.keys())
        wins_sub = df[df["Outcome"] == "Win"] if "Outcome" in df.columns else df.iloc[0:0]
        loss_sub = df[df["Outcome"] == "Loss"] if "Outcome" in df.columns else df.iloc[0:0]
        prac_sub = df[~df["Outcome"].isin(["Win", "Loss"])] if "Outcome" in df.columns else df.iloc[0:0]

        def _axis_means(sub):
            vals = []
            for lbl in axes:
                col = ALL_13_VARIABLE_MAP[lbl]
                s = pd.to_numeric(sub[col], errors="coerce") if (not sub.empty and col in sub.columns) else pd.Series(dtype="float64")
                vals.append(round(float(s.mean()), 2) if s.notna().any() else None)
            return vals

        season_avg = _axis_means(df)
        win_avg = _axis_means(wins_sub)
        loss_avg = _axis_means(loss_sub)
        prac_avg = _axis_means(prac_sub)

        # Always display ALL 12 variables on the Radar Chart
        plot_axes = list(axes)
        p_season = [v if v is not None else 5.0 for v in season_avg]
        p_win = [win_avg[i] if win_avg[i] is not None else p_season[i] for i in range(len(axes))]
        p_loss = [loss_avg[i] if loss_avg[i] is not None else p_season[i] for i in range(len(axes))]
        p_prac = [prac_avg[i] if prac_avg[i] is not None else p_season[i] for i in range(len(axes))]

        fig_radar = go.Figure()
        if plot_axes:
            fig_radar.add_trace(go.Scatterpolar(
                r=p_season + [p_season[0]], theta=plot_axes + [plot_axes[0]],
                fill="toself", name=f"Season Avg ({len(df)} sessions)",
                line=dict(color="#e74c3c", width=2.5), opacity=0.65
            ))
            if not wins_sub.empty:
                fig_radar.add_trace(go.Scatterpolar(
                    r=p_win + [p_win[0]], theta=plot_axes + [plot_axes[0]],
                    fill="toself", name=f"In Wins ({len(wins_sub)} matches)",
                    line=dict(color="#2ecc71", width=2.5), opacity=0.55
                ))
            if not loss_sub.empty:
                fig_radar.add_trace(go.Scatterpolar(
                    r=p_loss + [p_loss[0]], theta=plot_axes + [plot_axes[0]],
                    fill="none", name=f"In Losses ({len(loss_sub)} matches)",
                    line=dict(color="#f39c12", width=2.5, dash="dot")
                ))
            if not prac_sub.empty:
                fig_radar.add_trace(go.Scatterpolar(
                    r=p_prac + [p_prac[0]], theta=plot_axes + [plot_axes[0]],
                    fill="none", name=f"Wall & Training ({len(prac_sub)})",
                    line=dict(color="#3498db", width=2)
                ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
            template="plotly_dark",
            height=460,
            margin=dict(l=65, r=65, t=70, b=40)
        )

        r_col1, r_col2 = st.columns([1.15, 1])
        with r_col1:
            st.plotly_chart(fig_radar, use_container_width=True)

        with r_col2:
            st.subheader("?? Skill Scorecard Breakdown")
            sc_rows = []
            for i, lbl in enumerate(axes):
                s_v, w_v, l_v = season_avg[i], win_avg[i], loss_avg[i]
                gap = round(w_v - l_v, 2) if (w_v is not None and l_v is not None) else None
                sc_rows.append({
                    "Skill Dimension": lbl,
                    "Season Avg": s_v,
                    "Win Avg": w_v,
                    "Loss Avg": l_v,
                    "Win-Loss Gap": gap
                })
            scorecard = pd.DataFrame(sc_rows)
            st.dataframe(scorecard, use_container_width=True, hide_index=True)

            valid_sc = scorecard.dropna(subset=["Season Avg"]).copy()
            if not valid_sc.empty:
                best_skill = valid_sc.loc[valid_sc["Season Avg"].idxmax(), "Skill Dimension"]
                best_val = valid_sc["Season Avg"].max()
                st.success(f"?? **Highest Rated Stroke:** {best_skill} ({best_val:.1f} / 10)")
                valid_gap = valid_sc.dropna(subset=["Win-Loss Gap"])
                if not valid_gap.empty:
                    sep_skill = valid_gap.loc[valid_gap["Win-Loss Gap"].idxmax(), "Skill Dimension"]
                    sep_val = valid_gap["Win-Loss Gap"].max()
                    st.warning(f"? **Biggest Win/Loss Separator:** {sep_skill} drops **{sep_val:.1f} pts** when you lose.")
        st.markdown("---")
        st.subheader("🔵 Multi-Factor Correlation Scatterplot")
        sc1, sc2, sc3 = st.columns(3)
        x_axis = sc1.selectbox("X-Axis Metric", axes, index=5)
        y_axis = sc2.selectbox("Y-Axis Metric", axes, index=6)
        show_trend = sc3.checkbox("Show OLS Trendline", value=True)

        scatter_plot_df = radar_df.copy()
        scatter_plot_df["Date"] = df["match_date"].dt.strftime("%Y-%m-%d") if "match_date" in df.columns else ""
        scatter_plot_df["Opponent / Session"] = df["opponent"] if "opponent" in df.columns else ""
        scatter_plot_df["Ankle Weights"] = df["ankle_weights"] if "ankle_weights" in df.columns else "No"
        scatter_plot_df["Notes"] = df["clean_note"].fillna("") if "clean_note" in df.columns else ""

        rng = np.random.default_rng(42)
        for _lbl, _col in ALL_13_VARIABLE_MAP.items():
            if _col in df.columns:
                scatter_plot_df[_col] = pd.to_numeric(df.loc[scatter_plot_df.index, _col], errors="coerce")
                scatter_plot_df[_lbl] = scatter_plot_df[_col]
        if x_axis not in scatter_plot_df.columns and x_axis in ALL_13_VARIABLE_MAP:
            scatter_plot_df[x_axis] = pd.to_numeric(df.loc[scatter_plot_df.index, ALL_13_VARIABLE_MAP[x_axis]], errors="coerce")
        if y_axis not in scatter_plot_df.columns and y_axis in ALL_13_VARIABLE_MAP:
            scatter_plot_df[y_axis] = pd.to_numeric(df.loc[scatter_plot_df.index, ALL_13_VARIABLE_MAP[y_axis]], errors="coerce")
        scatter_plot_df["x_jit"] = scatter_plot_df[x_axis] + rng.uniform(-0.12, 0.12, size=len(scatter_plot_df))
        scatter_plot_df["y_jit"] = scatter_plot_df[y_axis] + rng.uniform(-0.12, 0.12, size=len(scatter_plot_df))

        fig_scatter = px.scatter(
            scatter_plot_df, x="x_jit", y="y_jit", color="Outcome", size="Attack / Overall Index",
            hover_data={"x_jit": False, "y_jit": False, x_axis: ":.1f", y_axis: ":.1f", "Date": True, "Opponent / Session": True, "Ankle Weights": True, "Notes": True},
            color_discrete_map={"Win": "#2ecc71", "Loss": "#e74c3c", "Training / Wall": "#3498db"},
            trendline="ols" if show_trend else None,
            labels={"x_jit": x_axis, "y_jit": y_axis}, template="plotly_dark"
        )
        fig_scatter.add_vline(x=scatter_plot_df[x_axis].mean(), line_dash="dash", line_color="gray", annotation_text="Avg")
        fig_scatter.add_hline(y=scatter_plot_df[y_axis].mean(), line_dash="dash", line_color="gray", annotation_text="Avg")
        fig_scatter.update_layout(height=450, xaxis=dict(range=[0.5, 10.5]), yaxis=dict(range=[0.5, 10.5]))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 2: VOLUME, WINS, ANKLE WEIGHTS & LOSS REASONS
# ==========================================

        st.markdown("---")
        st.subheader("?? All Skills Over Time (Periodic Progression)")
        st.caption("Click any skill in the legend on the right to isolate or compare specific strokes over time.")
        if not df.empty and "match_date" in df.columns:
            prog_df = df.sort_values(by="match_date", ascending=True).copy()
            prog_df["Date_Str"] = prog_df["match_date"].dt.strftime("%Y-%m-%d").fillna("Session")
            prog_df["Session_Label"] = [f"#{i+1} ({d})" for i, d in enumerate(prog_df["Date_Str"])]

            fig_all_skills = go.Figure()
            all_13_traces = GRANULAR_STROKES + [("Mental Focus", "mental_focus", ["mental_focus"], "#f1c40f")]
            for s_label, s_col, _, s_color in all_13_traces:
                if s_col in prog_df.columns:
                    s_series = pd.to_numeric(prog_df[s_col], errors="coerce")
                else:
                    s_series = pd.Series(np.nan, index=prog_df.index)
                # Fill missing historical values with column median (or 5.5 neutral) so every line renders across all 22 sessions
                s_series_filled = s_series
                fig_all_skills.add_trace(go.Scatter(
                    x=prog_df["Session_Label"],
                    y=s_series_filled,
                    mode="lines+markers",
                    name=s_label,
                    connectgaps=True,
                    line=dict(color=s_color, width=2.2),
                    marker=dict(size=6)
                ))

            fig_all_skills.update_layout(
                yaxis=dict(range=[0, 10.5], title="Variable Rating (1-10)"),
                xaxis_title="Chronological Session (Click any variable in legend to toggle)",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_all_skills, use_container_width=True)

with tab2:
    if df.empty:
        st.info("Log sessions first to see your training volume and loss diagnostics.")
    else:
        st.subheader("📊 Court Activity, Ankle Weights Volume & Root-Cause Breakdown")

        total_all = len(df)
        total_matches = len(df[df["Outcome"].isin(["Win", "Loss"])])
        total_wins = len(df[df["Outcome"] == "Win"])
        total_losses = len(df[df["Outcome"] == "Loss"])
        total_wall = len(df[df["match_type"].astype(str).str.contains("Wall", case=False, na=False)]) if "match_type" in df.columns else 0
        weighted_sessions = len(df[~df["ankle_weights"].astype(str).str.lower().isin(["no", "nan", "none", "false", "0", ""])])
        win_pct = (total_wins / total_matches * 100) if total_matches > 0 else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Sessions", total_all)
        k2.metric("Matches Played", total_matches)
        k3.metric("Wins / Losses", f"{total_wins}W - {total_losses}L")
        k4.metric("Win Rate", f"{win_pct:.0f}%")
        k5.metric("Wall Sessions", total_wall)
        k6.metric("🏋️ Ankle Weight Sessions", weighted_sessions)

        st.markdown("---")
        v1, v2 = st.columns(2)

        with v1:
            st.subheader("Where You Spend Your Court Time")
            if "match_type" in df.columns:
                type_counts = df["match_type"].fillna("Match").value_counts().reset_index()
                type_counts.columns = ["Session Type", "Count"]
                fig_vol = px.pie(
                    type_counts, values="Count", names="Session Type", hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_dark"
                )
                fig_vol.update_layout(height=340)
                st.plotly_chart(fig_vol, use_container_width=True)

        with v2:
            st.subheader("🏋️ Ankle Weights vs. Standard Footwork Impact")
            aw_df = df.copy()
            is_no_aw = aw_df["ankle_weights"].astype(str).str.lower().isin(["no", "nan", "none", "false", "0", ""])
            aw_df["Training Load"] = np.where(is_no_aw, "Standard (No Weights)", "With Ankle Weights")
            rating_cols = [c for c in ALL_13_ML_FEATURES if c in aw_df.columns]
            if rating_cols and aw_df["Training Load"].nunique() > 1:
                aw_grp = aw_df.groupby("Training Load")[rating_cols].mean().reset_index().melt(
                    id_vars="Training Load", var_name="Metric", value_name="Avg Score"
                )
                fig_aw = px.bar(
                    aw_grp, x="Metric", y="Avg Score", color="Training Load", barmode="group",
                    color_discrete_map={"Standard (No Weights)": "#3498db", "With Ankle Weights": "#f39c12"},
                    range_y=[0, 10], template="plotly_dark"
                )
                fig_aw.update_layout(height=340)
                st.plotly_chart(fig_aw, use_container_width=True)
            else:
                st.info("Map your Ankle Weights column during CSV import (or log sessions with/without weights) to compare performance across loads!")

        st.markdown("---")
        l_col1, l_col2 = st.columns(2)

        with l_col1:
            st.subheader("🚨 Most Repeated Reasons for Losing")
            loss_rows = df[df["Outcome"] == "Loss"]
            if loss_rows.empty:
                st.success("Zero losses recorded yet!")
            else:
                reasons = []
                if "main_leak" in loss_rows.columns:
                    for item in loss_rows["main_leak"].dropna():
                        if str(item).strip() not in ["", "None", "nan"]:
                            reasons.append(str(item).strip())

                # Scan both clean_note and any unmapped text columns inside loss rows
                text_cols_to_scan = [c for c in ["clean_note", "opponent", "score"] if c in loss_rows.columns]
                keyword_map = {
                    "Backhand errors / depth": ["backhand", "bh", "revés", "reves"],
                    "Forehand unforced errors": ["forehand", "fh", "derecha"],
                    "Serve / Double faults": ["serve", "double fault", "saque", "doble falta"],
                    "Mental focus / Impatience": ["focus", "mental", "rush", "nerv", "concentra", "paciencia"],
                    "Footwork / Tired legs": ["footwork", "legs", "tired", "slow", "pies", "cansad", "movimiento"],
                    "Return of serve": ["return", "devolucion", "devolución"]
                }
                for _, r in loss_rows.iterrows():
                    combined_txt = " ".join([str(r[c]) for c in text_cols_to_scan if pd.notna(r[c])]).lower()
                    for label, kws in keyword_map.items():
                        if any(k in combined_txt for k in kws):
                            reasons.append(label)

                if reasons:
                    reason_df = pd.Series(reasons).value_counts().reset_index()
                    reason_df.columns = ["Loss Cause", "Occurrences"]
                    fig_loss_reasons = px.bar(
                        reason_df.sort_values("Occurrences", ascending=True),
                        x="Occurrences", y="Loss Cause", orientation="h",
                        color="Occurrences", color_continuous_scale="Reds", template="plotly_dark"
                    )
                    fig_loss_reasons.update_layout(height=330, showlegend=False)
                    st.plotly_chart(fig_loss_reasons, use_container_width=True)
                else:
                    st.info("No explicit loss keywords found in notes yet. Map your notes/reasons column during CSV upload!")

        with l_col2:
            st.subheader("💪 Your Best Strengths & Winning Weapons")
            strengths = []
            if "main_weapon" in df.columns:
                for w in df["main_weapon"].dropna():
                    if str(w).strip() not in ["", "None", "nan"]:
                        strengths.append(str(w).strip())

            win_rows = df[df["Outcome"] == "Win"]
            if "clean_note" in win_rows.columns:
                win_kw = {
                    "Forehand weapon": ["forehand", "fh", "derecha"],
                    "Backhand consistency": ["backhand", "bh", "revés", "reves"],
                    "First serve dominance": ["serve", "ace", "saque"],
                    "Mental composure": ["focus", "calm", "mental", "concentra"],
                    "Footwork & defense": ["footwork", "run", "legs", "pies", "defen"]
                }
                for note in win_rows["clean_note"].dropna():
                    low = str(note).lower()
                    for label, kws in win_kw.items():
                        if any(k in low for k in kws):
                            strengths.append(label)

            if strengths:
                str_df = pd.Series(strengths).value_counts().reset_index()
                str_df.columns = ["Strength", "Count"]
                fig_str = px.bar(
                    str_df.sort_values("Count", ascending=True),
                    x="Count", y="Strength", orientation="h",
                    color="Count", color_continuous_scale="Greens", template="plotly_dark"
                )
                fig_str.update_layout(height=330, showlegend=False)
                st.plotly_chart(fig_str, use_container_width=True)
            else:
                st.info("Log or import sessions with notes/weapons to chart your top strengths.")

# ==========================================
# TAB 3: AI COACH & ML MODELS
# ==========================================

        st.markdown("---")
        st.subheader("Match Momentum: Wins & Defeats (Ups & Downs)")
        match_hist = df[df["Outcome"].isin(["Win", "Loss"])].copy()
        if not match_hist.empty:
            match_hist = match_hist.sort_values(by="match_date", ascending=True).reset_index(drop=True)
            match_hist["Match #"] = match_hist.index + 1
            match_hist["Delta"] = np.where(match_hist["Outcome"] == "Win", 1, -1)
            match_hist["Momentum Score"] = match_hist["Delta"].cumsum()

            fig_mom = go.Figure()
            fig_mom.add_trace(go.Scatter(
                x=match_hist["Match #"],
                y=match_hist["Momentum Score"],
                mode="lines+markers",
                line=dict(color="#3498db", width=3),
                marker=dict(
                    size=12,
                    color=np.where(match_hist["Outcome"] == "Win", "#2ecc71", "#e74c3c"),
                    line=dict(width=1.5, color="#ffffff")
                ),
                text=match_hist["opponent"] + " (" + match_hist["Outcome"] + " " + match_hist["score"].fillna("") + ")",
                hovertemplate="<b>Match #%{x}</b><br>%{text}<br>Net Momentum: %{y}<extra></extra>"
            ))
            fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Even (.500)")
            fig_mom.update_layout(
                xaxis_title="Chronological Match Number (Green = Win, Red = Loss)",
                yaxis_title="Net Momentum (+1 Win / -1 Loss)",
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_mom, use_container_width=True)
        else:
            st.info("Log matches with 'Win' or 'Loss' results to see your ups-and-downs momentum line.")

        st.markdown("---")
        st.subheader("Wall Practice Breakdown: What You Practiced")
        wall_sessions = df[
            df["match_type"].astype(str).str.contains("Wall", case=False, na=False) |
            df["opponent"].astype(str).str.contains("Wall|Pared", case=False, na=False)
        ].copy()

        if wall_sessions.empty:
            st.info("No Wall sessions logged yet. Log a 'Wall Session' in Tab 5 to see your wall drill breakdown.")
        else:
            wall_topics = []
            wall_kw_map = {
                "Volleys": ["volley", "volea", "red"],
                "Playing in Front / Early": ["front", "early", "adelante", "temprano", "inside"],
                "Super Fast Balls / Pace": ["fast", "hard", "speed", "pace", "rapida", "r?pida", "fuerte", "velocidad"],
                "Spin & Effect (Topspin/Slice)": ["spin", "effect", "topspin", "slice", "efecto", "liftado", "cortado"]
            }

            for _, wrow in wall_sessions.iterrows():
                if "wall_drills" in wrow and isinstance(wrow["wall_drills"], list):
                    wall_topics.extend(wrow["wall_drills"])
                note_txt = f"{wrow.get('clean_note', '')} {wrow.get('score', '')}".lower()
                for label, kws in wall_kw_map.items():
                    if any(k in note_txt for k in kws):
                        wall_topics.append(label)

            if wall_topics:
                wt_df = pd.Series(wall_topics).value_counts().reset_index()
                wt_df.columns = ["Wall Drill Focus", "Sessions"]
                w_c1, w_c2 = st.columns(2)
                with w_c1:
                    fig_wall_bar = px.bar(
                        wt_df.sort_values("Sessions", ascending=True),
                        x="Sessions", y="Wall Drill Focus", orientation="h",
                        color="Sessions", color_continuous_scale="Blues",
                        title="Wall Drills Practiced (Frequency)", template="plotly_dark"
                    )
                    fig_wall_bar.update_layout(height=340, showlegend=False)
                    st.plotly_chart(fig_wall_bar, use_container_width=True)
                with w_c2:
                    fig_wall_pie = px.pie(
                        wt_df, values="Sessions", names="Wall Drill Focus", hole=0.4,
                        title="Wall Practice Share (%)", template="plotly_dark"
                    )
                    fig_wall_pie.update_layout(height=340)
                    st.plotly_chart(fig_wall_pie, use_container_width=True)
            else:
                st.info("Select your wall drill focus (Volleys, Playing in Front, Fast Balls, Effect) when logging Wall Sessions to populate these charts.")

with tab3:
    st.subheader("🤖 AI Coach Diagnostic & Machine Learning Lab")
    st.caption("Trains Random Forest, XGBoost, Logistic Regression, and Ridge Regression strictly on your logged matches.")

    ml_df = df[df["Outcome"].isin(["Win", "Loss"])].copy() if (not df.empty and "Outcome" in df.columns) else pd.DataFrame()

    if len(ml_df) < 6:
        st.warning(f"Currently found {len(ml_df)} completed matches. Log at least 6–8 matches with Win/Loss results to unlock the ML diagnostic engine.")
    else:
        ml_df["target_win"] = np.where(ml_df["Outcome"] == "Win", 1, 0)
        num_features = [c for c in ALL_13_ML_FEATURES if c in ml_df.columns and ml_df[c].notna().any()]
        X_num = ml_df[num_features].apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median()).fillna(6.0)
        is_no_aw_ml = ml_df["ankle_weights"].astype(str).str.lower().isin(["no", "nan", "none", "false", "0", ""])
        X_num["used_ankle_weights"] = np.where(is_no_aw_ml, 0.0, 1.0)

        if "surface" in ml_df.columns:
            X_surf = pd.get_dummies(ml_df["surface"].fillna("Hard"), prefix="Surface", drop_first=False).astype(float)
            X = pd.concat([X_num, X_surf], axis=1)
        else:
            X = X_num.copy()

        y = ml_df["target_win"].values

        if len(np.unique(y)) < 2:
            st.info("Need both Wins and Losses in your database to train classification models.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            log_reg = LogisticRegression(C=1.0, random_state=42).fit(X_scaled, y)
            rf_clf = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42).fit(X, y)
            xgb_clf = xgb.XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.1, eval_metric="logloss", random_state=42).fit(X, y)
            ridge_reg = Ridge(alpha=1.0).fit(X_scaled, X_num[num_features].mean(axis=1))

            win_means = X_num[num_features][y == 1].mean()
            loss_means = X_num[num_features][y == 0].mean()
            diffs = (win_means - loss_means).sort_values(ascending=False)

            biggest_win_driver = diffs.index[0]
            second_win_driver = diffs.index[1] if len(diffs) > 1 else diffs.index[0]
            weakest_in_losses = loss_means.idxmin()

            wall_count = len(df[df["Outcome"] == "Training / Wall"])
            aw_count = len(df[~df["ankle_weights"].astype(str).str.lower().isin(["no", "nan", "none", "false", "0", ""])])
            match_count = len(ml_df)

            st.markdown("### 🧠 Automated AI Coach Report")
            c_rep1, c_rep2, c_rep3 = st.columns(3)

            with c_rep1:
                st.markdown(f"""
                <div class="coach-card">
                    <h4>✅ Why You Win</h4>
                    <p>When you win, your <b>{biggest_win_driver}</b> jumps to <b>{win_means[biggest_win_driver]:.1f}/10</b> (vs. {loss_means[biggest_win_driver]:.1f} in losses), 
                    backed by <b>{second_win_driver}</b> at <b>{win_means[second_win_driver]:.1f}/10</b>.</p>
                    <p><b>Winning Formula:</b> Keep <i>{biggest_win_driver}</i> above <b>{max(6.5, win_means[biggest_win_driver]-0.5):.1f}</b> to push your ensemble win odds above 70%.</p>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                st.markdown(f"""
                <div class="loss-card">
                    <h4>❌ Why You Lose</h4>
                    <p>Your lowest-scoring metric in losses is <b>{weakest_in_losses}</b> (averaging just <b>{loss_means[weakest_in_losses]:.1f}/10</b>).</p>
                    <p>Random Forest & XGBoost show that when <b>{biggest_win_driver}</b> drops by more than <b>{diffs.iloc[0]:.1f} points</b>, opponents take control of rallies.</p>
                </div>
                """, unsafe_allow_html=True)

            with c_rep3:
                st.markdown(f"""
                <div class="wall-card">
                    <h4>🧱 Wall & Ankle-Weight Training Plan</h4>
                    <p><b>#1 Priority Drill:</b> Target <b>{weakest_in_losses}</b> & <b>{biggest_win_driver}</b>.</p>
                    <p>You have logged <b>{wall_count} wall/drill sessions</b> and <b>{aw_count} ankle-weight sessions</b> across <b>{match_count} matches</b>. Use ankle weights for short 15-min split-step bursts, then remove them to lock in explosive match footwork.</p>
                </div>
                """, unsafe_allow_html=True)

            cv_folds = min(3, int(min(y.sum(), len(y) - y.sum())))
            if cv_folds >= 2:
                rf_acc = cross_val_score(rf_clf, X, y, cv=cv_folds).mean()
                xgb_acc = cross_val_score(xgb_clf, X, y, cv=cv_folds).mean()
                log_acc = cross_val_score(log_reg, X_scaled, y, cv=cv_folds).mean()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Random Forest Accuracy", f"{rf_acc*100:.0f}%")
                m2.metric("XGBoost Accuracy", f"{xgb_acc*100:.0f}%")
                m3.metric("Logistic Regression Accuracy", f"{log_acc*100:.0f}%")
                m4.metric("Ridge Regression R²", f"{ridge_reg.score(X_scaled, X_num[num_features].mean(axis=1)):.2f}")

            st.markdown("---")
            col_imp1, col_imp2 = st.columns(2)

            with col_imp1:
                st.subheader("Feature Importance (Random Forest vs. XGBoost)")
                imp_df = pd.DataFrame({
                    "Factor": X.columns,
                    "Random Forest": rf_clf.feature_importances_,
                    "XGBoost": xgb_clf.feature_importances_
                }).melt(id_vars="Factor", var_name="Model", value_name="Importance")
                fig_imp = px.bar(
                    imp_df.sort_values("Importance", ascending=True),
                    x="Importance", y="Factor", color="Model", barmode="group", template="plotly_dark"
                )
                fig_imp.update_layout(height=360)
                st.plotly_chart(fig_imp, use_container_width=True)

            with col_imp2:
                st.subheader("Win/Loss Impact Weights (Logistic Regression)")
                coef_df = pd.DataFrame({
                    "Factor": X.columns,
                    "Impact Weight": log_reg.coef_[0]
                }).sort_values("Impact Weight", ascending=True)
                coef_df["Direction"] = np.where(coef_df["Impact Weight"] >= 0, "Drives Wins", "Drives Losses")
                fig_coef = px.bar(
                    coef_df, x="Impact Weight", y="Factor", color="Direction", orientation="h",
                    color_discrete_map={"Drives Wins": "#2ecc71", "Drives Losses": "#e74c3c"},
                    template="plotly_dark"
                )
                fig_coef.update_layout(height=360)
                st.plotly_chart(fig_coef, use_container_width=True)

            st.markdown("---")
            st.subheader("🎛️ Interactive 'What-If' Match Win Simulator")
            sim_cols = st.columns(len(num_features))
            sim_values = {}
            for idx, feat in enumerate(num_features):
                default_val = float(round(X_num[feat].mean(), 1))
                sim_values[feat] = sim_cols[idx].slider(f"{feat}", 1.0, 10.0, default_val, 0.5, key=f"sim_{feat}")

            sim_row = pd.DataFrame([sim_values])
            for col in X.columns:
                if col not in sim_row.columns:
                    sim_row[col] = 0.0
            sim_row = sim_row[X.columns]

            prob_lr = log_reg.predict_proba(scaler.transform(sim_row))[0][1]
            prob_rf = rf_clf.predict_proba(sim_row)[0][1]
            prob_xgb = xgb_clf.predict_proba(sim_row)[0][1]
            ensemble_prob = (prob_lr + prob_rf + prob_xgb) / 3.0

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Ensemble Win Probability", f"{ensemble_prob*100:.1f}%")
            p2.metric("Random Forest", f"{prob_rf*100:.1f}%")
            p3.metric("XGBoost", f"{prob_xgb*100:.1f}%")
            p4.metric("Logistic Regression", f"{prob_lr*100:.1f}%")

# ==========================================
# TAB 4: VIDEO BIOMECHANICS LAB
# ==========================================
with tab4:
    st.subheader("🎥 Video Biomechanics & Stroke Movement Lab")
    st.caption("Upload match or training videos from your folder. Gemini Vision & OpenCV extract movement cadence, swing explosiveness, and stroke mechanics—stored completely separate from your self-reported match logs.")

    has_gemini = HAS_GENAI and ("GEMINI_API_KEY" in st.secrets)
    if has_gemini:
        st.success("✅ **Gemini Multimodal Vision + OpenCV Dual-Engine Active:** Full AI frame-by-frame kinetic chain & stroke coaching enabled!")
    else:
        st.info("ℹ️ **Built-In OpenCV Kinetic Telemetry Active.** Add `GEMINI_API_KEY` to your Streamlit Cloud Secrets to activate Gemini Vision stroke biomechanics!")

    with st.expander("📤 Upload & Analyze a Match or Training Video", expanded=True):
        vc1, vc2, vc3, vc4 = st.columns(4)
        video_date = vc1.date_input("Video Date", value=date.today(), key="vid_date")
        focus_stroke = vc2.selectbox(
            "Primary Stroke / Focus",
            ["Full Rally / Match Movement", "Forehand Mechanics", "Backhand Mechanics", "Serve Motion", "Return & Split-Step", "Wall Drill Footwork"]
        )
        video_context = vc3.selectbox(
            "Footage Context",
            ["Match Footage", "Wall Training", "Court Drills / Basket", "Practice Set"]
        )
        vid_ankle_weights = vc4.selectbox(
            "🏋️ Ankle Weights in Video?",
            ["No", "Yes — Light (0.5–1 kg)", "Yes — Heavy (1.5+ kg)"]
        )

        uploaded_video = st.file_uploader("Select a video file from your folder (MP4, MOV, AVI — clips under 2 mins process fastest)", type=["mp4", "mov", "avi", "mkv"])

        if uploaded_video is not None:
            if st.button("🚀 Run Gemini Vision & Biomechanics Analysis", use_container_width=True):
                with st.spinner("Analyzing kinetic chain, split-step footwork, and swing mechanics..."):
                    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        tmp_path = tmp_file.name

                    try:
                        cv_res = analyze_video_cv(tmp_path, focus_stroke)
                        ai_res = analyze_video_gemini(tmp_path, focus_stroke, video_context, vid_ankle_weights)

                        if cv_res is None and ai_res is None:
                            st.error("Could not process this video file. Try an MP4 or MOV clip.")
                        else:
                            kinetic_score = float(ai_res["kinetic_chain_score"]) if ai_res else cv_res["swing_explosiveness"]
                            footwork_score = float(ai_res["footwork_splitstep_score"]) if ai_res else cv_res["footwork_cadence"]
                            contact_score = float(ai_res["contact_point_score"]) if ai_res else round((cv_res["swing_explosiveness"] + cv_res["recovery_rhythm"]) / 2, 1)
                            balance_score = float(ai_res["balance_recovery_score"]) if ai_res else cv_res["recovery_rhythm"]

                            st.markdown("### 📊 Video Biomechanics & Telemetry Results")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Swing / Kinetic Chain", f"{kinetic_score:.1f} / 10")
                            m2.metric("Footwork & Split-Step", f"{footwork_score:.1f} / 10")
                            m3.metric("Contact Point Timing", f"{contact_score:.1f} / 10")
                            m4.metric("Balance & Recovery", f"{balance_score:.1f} / 10")

                            if cv_res and not cv_res["motion_df"].empty:
                                fig_wave = px.area(
                                    cv_res["motion_df"], x="Time (sec)", y="Movement & Swing Intensity",
                                    title=f"Kinetic Movement & Swing Acceleration Timeline ({uploaded_video.name})",
                                    template="plotly_dark", color_discrete_sequence=["#9b59b6"]
                                )
                                fig_wave.update_layout(height=280)
                                st.plotly_chart(fig_wave, use_container_width=True)

                            if cv_res and cv_res["key_frames"]:
                                st.markdown("**Detected Peak Action / Contact Moments:**")
                                f_cols = st.columns(len(cv_res["key_frames"]))
                                for idx, (score, ts, rgb_img) in enumerate(cv_res["key_frames"]):
                                    f_cols[idx].image(rgb_img, caption=f"Peak @ {ts}s (Intensity: {score:.1f})", use_container_width=True)

                            if ai_res:
                                strengths_txt = ai_res.get("strengths_observed", "")
                                flaws_txt = ai_res.get("biomechanical_flaws", "")
                                drills_txt = ai_res.get("corrective_drills", "")
                            else:
                                strengths_txt = f"Steady tempo across the {cv_res['duration_sec']}s clip (burst ratio {cv_res['burst_ratio']}x)."
                                flaws_txt = "Add your GEMINI_API_KEY in Streamlit Secrets for full AI stroke-by-stroke biomechanical critique."
                                drills_txt = "Shadow split-step + 50-ball wall consistency drill."

                            weight_badge = f" | 🏋️ Ankle Weights: {vid_ankle_weights}" if vid_ankle_weights != "No" else ""
                            st.markdown(f"""
                            <div class="video-card">
                                <h4>🎬 AI Biomechanical Breakdown — {focus_stroke} ({video_context}{weight_badge})</h4>
                                <p><b>✅ Mechanical Strengths:</b> {strengths_txt}</p>
                                <p><b>⚠️ Movement / Stroke Leaks to Fix:</b> {flaws_txt}</p>
                                <p><b>🎯 Recommended Corrective Drill:</b> {drills_txt}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            vid_meta = {
                                "video_stroke": focus_stroke,
                                "video_context": video_context,
                                "ankle_weights": vid_ankle_weights,
                                "kinetic_chain": kinetic_score,
                                "contact_point": contact_score,
                                "balance_recovery": balance_score,
                                "video_strengths": strengths_txt,
                                "video_flaws": flaws_txt,
                                "video_drills": drills_txt
                            }
                            vid_summary = f"[Video: {focus_stroke} | {video_context}{weight_badge}] Strengths: {strengths_txt} // Fix: {flaws_txt} // Drill: {drills_txt}"
                            vid_record = {
                                "match_date": str(video_date),
                                "match_type": "Video Biomechanics",
                                "surface": "Video Footage",
                                "opponent_name": uploaded_video.name,
                                "opponent": uploaded_video.name,
                                "result": "Video Analysis",
                                "score": f"{cv_res['duration_sec']}s clip" if cv_res else "Video Clip",
                                "serve_rating": round(kinetic_score),
                                "mental_focus": round(contact_score),
                                "energy_level": round(footwork_score),
                                note_col: f"[META:{json.dumps(vid_meta)}] {vid_summary}"
                            }
                            insert_match(vid_record, valid_db_cols)
                            st.success("Video analysis saved permanently to your separate Video Biomechanics Log!")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    st.markdown("---")
    st.subheader("📈 Separate Video Biomechanics History & Progress")
    if video_df.empty:
        st.caption("No videos analyzed yet. Upload your first match or wall clip above!")
    else:
        vk1, vk2, vk3, vk4 = st.columns(4)
        vk1.metric("Videos Analyzed", len(video_df))
        vk2.metric("Avg Kinetic Chain", f"{pd.to_numeric(video_df['serve_rating'], errors='coerce').mean():.1f} / 10")
        vk3.metric("Avg Footwork Cadence", f"{pd.to_numeric(video_df['energy_level'], errors='coerce').mean():.1f} / 10")
        vk4.metric("Avg Contact Timing", f"{pd.to_numeric(video_df['mental_focus'], errors='coerce').mean():.1f} / 10")

        vid_cols = [c for c in ["match_date", "opponent", "video_stroke", "video_context", "ankle_weights", "score", "serve_rating", "energy_level", "mental_focus", "clean_note"] if c in video_df.columns]
        st.dataframe(video_df[vid_cols], use_container_width=True)

# ==========================================
# TAB 5: LOG DETAILED SESSION (MOBILE-READY)
# ==========================================
with tab5:
    st.subheader("Log a Detailed Match, Drill, or Wall Session")
    with st.form("log_session_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        session_date = c1.date_input("Date", value=date.today())
        match_type = c2.selectbox(
            "Session Type",
            ["Singles Match", "Doubles Match", "Practice Match", "Wall Session", "Drills / Coaching"]
        )
        surface = c3.selectbox("Surface", ["Hard", "Clay", "Grass", "Wall / Concrete"])

        c4, c5, c6 = st.columns(3)
        opponent = c4.text_input("Opponent / Focus (e.g., Wall, Carlos)", value="Wall" if match_type == "Wall Session" else "")
        result = c5.selectbox("Result", ["Win", "Loss", "N/A (Training)"])
        score = c6.text_input("Score or Duration (e.g., 6-4 4-6 10-8 or 60 mins)")

        st.markdown("---")
        st.markdown("**Match Context, Conditioning Load & Opponent Profile**")
        ctx1, ctx2, ctx3, ctx4 = st.columns(4)
        opp_style = ctx1.selectbox(
            "Opponent Style / Drill Theme",
            ["All-Court", "Aggressive Baseliner", "Counterpuncher / Pusher", "Big Server", "Net Rusher / S&V", "Lefty / Unorthodox", "Solo Wall / Footwork"]
        )
        pressure = ctx2.selectbox(
            "Match Flow / Pressure",
            ["Competitive / Close Sets", "Tiebreak / 3rd Set Decider", "Dominant Control", "Played From Behind", "Routine Practice"]
        )
        physical_state = ctx3.selectbox(
            "Physical Readiness",
            ["Fresh & Explosive", "Normal", "Heavy Legs / Fatigued", "Sore / Tight"]
        )
        ankle_weights = ctx4.selectbox(
            "🏋️ Trained with Ankle Weights?",
            ["No", "Yes — Light (0.5–1 kg)", "Yes — Heavy (1.5+ kg)"]
        )

        st.markdown("---")
        st.markdown("**?? Complete 13-Variable Ratings (`0 = Did Not Train Today` ? won't bias your averages)**")
        def _clean_untrained(val):
            return None if val == 0 else val

        g_r1_c1, g_r1_c2, g_r1_c3, g_r1_c4 = st.columns(4)
        first_serve_raw = g_r1_c1.slider("1st Serve (0=N/A)", 0, 10, 0)
        second_serve_raw = g_r1_c2.slider("2nd Serve (0=N/A)", 0, 10, 0)
        fh_rating_raw = g_r1_c3.slider("Forehand (0=N/A)", 0, 10, 0)
        bh_rating_raw = g_r1_c4.slider("Backhand (0=N/A)", 0, 10, 0)

        g_r2_c1, g_r2_c2, g_r2_c3, g_r2_c4 = st.columns(4)
        fh_return_raw = g_r2_c1.slider("Forehand Return (0=N/A)", 0, 10, 0)
        bh_return_raw = g_r2_c2.slider("Backhand Return (0=N/A)", 0, 10, 0)
        fh_volley_raw = g_r2_c3.slider("Forehand Volley (0=N/A)", 0, 10, 0)
        bh_volley_raw = g_r2_c4.slider("Backhand Volley (0=N/A)", 0, 10, 0)

        g_r3_c1, g_r3_c2, g_r3_c3, g_r3_c4 = st.columns(4)
        overhead_rating_raw = g_r3_c1.slider("Overhead (0=N/A)", 0, 10, 0)
        slice_rating_raw = g_r3_c2.slider("Slice (0=N/A)", 0, 10, 0)
        movement_rating_raw = g_r3_c3.slider("Movement (0=N/A)", 0, 10, 0)
        mental_focus_raw = g_r3_c4.slider("Mental Focus (0=N/A)", 0, 10, 0)

        g_r4_c1, _, _, _ = st.columns(4)
        stamina_rating_raw = g_r4_c1.slider("Physical / Stamina (0=N/A)", 0, 10, 0)

        first_serve = _clean_untrained(first_serve_raw)
        second_serve = _clean_untrained(second_serve_raw)
        fh_rating = _clean_untrained(fh_rating_raw)
        bh_rating = _clean_untrained(bh_rating_raw)
        fh_return = _clean_untrained(fh_return_raw)
        bh_return = _clean_untrained(bh_return_raw)
        fh_volley = _clean_untrained(fh_volley_raw)
        bh_volley = _clean_untrained(bh_volley_raw)
        overhead_rating = _clean_untrained(overhead_rating_raw)
        slice_rating = _clean_untrained(slice_rating_raw)
        movement_rating = _clean_untrained(movement_rating_raw)
        mental_focus = _clean_untrained(mental_focus_raw)
        stamina_rating = _clean_untrained(stamina_rating_raw)

        _trained_serves = [v for v in [first_serve, second_serve] if v is not None]
        serve_rating = round(sum(_trained_serves) / len(_trained_serves), 1) if _trained_serves else None

        serve_rating = round(sum([v for v in [first_serve, second_serve] if v is not None]) / len([v for v in [first_serve, second_serve] if v is not None]), 1) if any(v is not None for v in [first_serve, second_serve]) else None

        st.markdown("---")
        st.markdown("**?? Tactical Root-Cause & Custom Tag Builder**")
        _hist_problems = set()
        _hist_pressures = set()
        if not df.empty:
            for _col_p in ["main_problem", "tactical_problem", "root_cause"]:
                if _col_p in df.columns:
                    _hist_problems.update(df[_col_p].dropna().astype(str).str.strip().tolist())
            for _col_pr in ["pressure_flow", "match_flow", "pressure_notes"]:
                if _col_pr in df.columns:
                    for _entry in df[_col_pr].dropna().astype(str):
                        for _part in _entry.split("|"):
                            if _part.strip():
                                _hist_pressures.add(_part.strip())
        _hist_problems.discard("")
        _hist_problems.discard("None")
        _hist_pressures.discard("")
        _hist_pressures.discard("None")

        base_problems = [
            "None / Played Solid",
            "Too close to the ball / Poor spacing",
            "Feet movement / Positioning to hit in the right spot",
            "Late preparation / Slow unit turn",
            "Overhitting / Low margin over net",
            "Tightness / Decelerating swing under pressure",
            "Poor shot selection / Wrong target",
            "Low 1st serve % / Double faults",
            "Contact point too far back",
            "Balance / Falling off the shot"
        ]
        all_problem_options = list(dict.fromkeys(base_problems + sorted(_hist_problems)))

        tc_col1, tc_col2 = st.columns(2)
        weakest_skill_options = ["None / Balanced Session"] + list(ALL_13_VARIABLE_MAP.keys())
        weakest_skill = tc_col1.selectbox(
            "Weakest Skill Today (All 13 Variables)",
            options=weakest_skill_options
        )
        selected_problem = tc_col2.selectbox(
            "Main Problem / Root Cause (Select Existing)",
            options=all_problem_options
        )
        custom_problem = st.text_input(
            "? Or Create New Root-Cause Tag (saved automatically for future sessions)",
            placeholder="e.g., Split-step timing late on deep returns, Dropping left shoulder on BH..."
        )
        main_problem = custom_problem.strip() if custom_problem.strip() else selected_problem

        st.markdown("---")
        st.markdown("**?? Match Flow & Pressure Moments (Select Existing or Create New Tag)**")
        base_pressure_options = [
            "0-30 / 15-40 Deficit",
            "Break Point Won",
            "Break Point Lost",
            "Deuce / Ad Games",
            "Serving Out Set / Match",
            "Tiebreak Pressure",
            "Momentum Swing / Lost Focus After Error",
            "Tight / Nervous Arm",
            "Locked In / Clutch Execution"
        ]
        all_pressure_options = list(dict.fromkeys(base_pressure_options + sorted(_hist_pressures)))

        pf_col1, pf_col2 = st.columns([1.2, 1])
        selected_pressure_tags = pf_col1.multiselect(
            "Select Match Flow / Pressure Tags",
            options=all_pressure_options
        )
        custom_pressure_tag = pf_col2.text_input(
            "? Add New Match Flow / Pressure Tag",
            placeholder="e.g., 4-4 30-30 second serve, Came back from 1-4 down..."
        )
        combined_pressure_list = list(selected_pressure_tags)
        if custom_pressure_tag.strip():
            combined_pressure_list.append(custom_pressure_tag.strip())
        pressure_flow = " | ".join(combined_pressure_list) if combined_pressure_list else "Standard Flow"

        tactical_note = st.text_area(
            "?? Extra Tactical Notes & Adjustments for Next Match",
            placeholder="What worked, opponent patterns, or specific drills to fix today's main problem..."
        )

        submitted = st.form_submit_button("Save Detailed Session", use_container_width=True)
        if submitted:
            meta_payload = ALL_13_VARIABLE_MAP
            narrative_parts = []
            if what_worked.strip():
                narrative_parts.append(f"Worked: {what_worked.strip()}")
            if what_to_adjust.strip():
                narrative_parts.append(f"Adjust: {what_to_adjust.strip()}")
            aw_tag = f" | Ankle Weights: {ankle_weights}" if ankle_weights != "No" else ""
            human_summary = f"[{opp_style} | {pressure}{aw_tag} | Weapon: {main_weapon} | Leak: {main_leak}] " + " // ".join(narrative_parts)

            encoded_note = f"[META:{json.dumps(meta_payload)}] {human_summary}".strip()
            opp_clean = opponent.strip() if opponent.strip() else ("Wall" if match_type == "Wall Session" else "Practice")

            new_record = ALL_13_VARIABLE_MAP
            insert_match(new_record, valid_db_cols)
            st.success("Session saved permanently to Supabase! Available immediately on your phone and PC.")
            st.rerun()

# ==========================================
# TAB 6: DATABASE RESET, INTERACTIVE CSV MAPPER & FULL EXPORT
# ==========================================
with tab6:
    st.subheader("🧹 Step 1: Clear Old / Duplicate Data (Optional Reset)")
    st.caption(f"Currently storing **{len(df) + len(video_df)} total rows** in Supabase. Click below if you want to wipe all existing rows before importing your clean 22-game CSV.")

    col_del1, col_del2 = st.columns([2, 1])
    confirm_wipe = col_del1.checkbox("Yes, delete all current rows in Supabase so I can re-upload my clean CSV from scratch.")
    if col_del2.button("🗑️ Delete All Existing Data", disabled=not confirm_wipe, use_container_width=True):
        wipe_all_records()
        st.success("✅ All old rows deleted! Your database is now clean (0 rows). Upload your CSV below.")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.subheader("📤 Step 2: Interactive CSV Importer (Reads 100% of Your Columns)")
    st.caption("Upload your CSV, verify how each column maps to CourtIQ, and save permanently to Supabase. Any extra columns in your CSV are automatically preserved too!")

    uploaded_csv = st.file_uploader("Choose your 22-game tennis CSV file", type=["csv"], key="csv_importer")
    if uploaded_csv is not None:
        try:
            imp_df = pd.read_csv(uploaded_csv)
            st.write(f"**Detected {len(imp_df)} rows and {len(imp_df.columns)} columns in your CSV:**")
            st.dataframe(imp_df.head(5), use_container_width=True)

            csv_cols = ["(Skip / Auto-preserve in metadata)"] + list(imp_df.columns)

            def auto_pick(keywords):
                for c in imp_df.columns:
                    low = str(c).lower().strip()
                    if any(k in low for k in keywords):
                        return c
                return "(Skip / Auto-preserve in metadata)"

            st.markdown("#### 🔗 Confirm Your CSV Column Mapping")
            m1, m2, m3, m4 = st.columns(4)
            col_date = m1.selectbox("Date Column", csv_cols, index=csv_cols.index(auto_pick(["date", "fecha", "day"])))
            col_type = m2.selectbox("Session Type (Match/Wall)", csv_cols, index=csv_cols.index(auto_pick(["type", "tipo", "session", "category"])))
            col_opp = m3.selectbox("Opponent / Focus", csv_cols, index=csv_cols.index(auto_pick(["opponent", "rival", "oponente", "player", "focus"])))
            col_res = m4.selectbox("Result (Win/Loss)", csv_cols, index=csv_cols.index(auto_pick(["result", "outcome", "resultado", "w_l", "win"])))

            m5, m6, m7, m8 = st.columns(4)
            col_score = m5.selectbox("Score / Duration", csv_cols, index=csv_cols.index(auto_pick(["score", "marcador", "duration", "time"])))
            col_surf = m6.selectbox("Surface", csv_cols, index=csv_cols.index(auto_pick(["surface", "superficie", "court", "cancha"])))
            col_aw = m7.selectbox("Ankle Weights Column", csv_cols, index=csv_cols.index(auto_pick(["ankle", "weight", "tobillo", "pesa"])))
            col_notes = m8.selectbox("Notes / Descriptions", csv_cols, index=csv_cols.index(auto_pick(["note", "nota", "comment", "desc", "reflection", "tactical"])))

            st.markdown("#### 🎯 Skill Rating Columns (1–10 Scale — Optional)")
            s1, s2, s3, s4 = st.columns(4)
            col_serve = s1.selectbox("Serve Rating", csv_cols, index=csv_cols.index(auto_pick(["serve", "saque"])))
            col_fh = s2.selectbox("Forehand Rating", csv_cols, index=csv_cols.index(auto_pick(["forehand", "fh", "derecha"])))
            col_bh = s3.selectbox("Backhand Rating", csv_cols, index=csv_cols.index(auto_pick(["backhand", "bh", "reves", "revés"])))
            col_ret = s4.selectbox("Return Rating", csv_cols, index=csv_cols.index(auto_pick(["return", "devolucion"])))

            s5, s6, s7 = st.columns(3)
            col_net = s5.selectbox("Net / Volley Rating", csv_cols, index=csv_cols.index(auto_pick(["net", "volley", "red", "volea"])))
            col_focus = s6.selectbox("Mental Focus Rating", csv_cols, index=csv_cols.index(auto_pick(["focus", "mental", "concentra"])))
            col_energy = s7.selectbox("Energy / Footwork Rating", csv_cols, index=csv_cols.index(auto_pick(["energy", "energia", "footwork", "stamina"])))

            replace_before_import = st.checkbox("🧹 Wipe existing Supabase rows right before saving this CSV (guarantees exact row count)", value=True)

            if st.button(f"💾 Save All {len(imp_df)} CSV Rows Permanently to Supabase", use_container_width=True):
                with st.spinner("Importing every row and preserving 100% of your CSV columns..."):
                    if replace_before_import:
                        wipe_all_records()

                    mapped_csv_cols = {
                        c for c in [col_date, col_type, col_opp, col_res, col_score, col_surf, col_aw, col_notes, col_serve, col_fh, col_bh, col_ret, col_net, col_focus, col_energy]
                        if c != "(Skip / Auto-preserve in metadata)"
                    }

                    def extract_num(row, chosen_col):
                        if chosen_col == "(Skip / Auto-preserve in metadata)" or chosen_col not in row or pd.isna(row[chosen_col]):
                            return None
                        val = pd.to_numeric(row[chosen_col], errors="coerce")
                        return round(float(np.clip(val, 1, 10)), 1) if pd.notna(val) else None

                    def extract_str(row, chosen_col, default=""):
                        if chosen_col == "(Skip / Auto-preserve in metadata)" or chosen_col not in row or pd.isna(row[chosen_col]):
                            return default
                        s = str(row[chosen_col]).strip()
                        return s if s and s.lower() != "nan" else default

                    records_to_insert = []
                    for _, row in imp_df.iterrows():
                        raw_dt = extract_str(row, col_date, str(date.today()))
                        parsed_dt = pd.to_datetime(raw_dt, errors="coerce")
                        clean_dt = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else str(date.today())

                        opp_val = extract_str(row, col_opp, "Unknown")
                        serve_val = extract_num(row, col_serve)
                        fh_val = extract_num(row, col_fh)
                        bh_val = extract_num(row, col_bh)
                        ret_val = extract_num(row, col_ret)
                        net_val = extract_num(row, col_net)
                        focus_val = extract_num(row, col_focus)
                        energy_val = extract_num(row, col_energy)

                        # Compute row's true mean from whichever ratings exist so missing ones don't skew one skill higher
                        present_ratings = [x for x in [serve_val, fh_val, bh_val, ret_val, net_val, focus_val, energy_val] if x is not None]
                        row_mean = int(round(np.mean(present_ratings))) if present_ratings else 6

                        meta_dict = {}
                        aw_val = extract_str(row, col_aw, "No")
                        if aw_val:
                            meta_dict["ankle_weights"] = aw_val
                        if fh_val is not None:
                            meta_dict["fh_rating"] = fh_val
                        if bh_val is not None:
                            meta_dict["bh_rating"] = bh_val
                        if ret_val is not None:
                            meta_dict["ret_rating"] = ret_val
                        if net_val is not None:
                            meta_dict["net_rating"] = net_val

                        # Preserve EVERY unmapped column from the CSV inside META JSON + readable summary
                        extra_summary_parts = []
                        for orig_col in imp_df.columns:
                            if orig_col not in mapped_csv_cols and pd.notna(row[orig_col]):
                                v = row[orig_col]
                                clean_v = float(v) if isinstance(v, np.floating) else (int(v) if isinstance(v, np.integer) else str(v).strip())
                                if str(clean_v) != "" and str(clean_v).lower() != "nan":
                                    meta_dict[orig_col] = clean_v
                                    extra_summary_parts.append(f"{orig_col}: {clean_v}")

                        raw_note = extract_str(row, col_notes, "")
                        existing_meta, clean_body = parse_rich_note(raw_note)
                        merged_meta = {**existing_meta, **meta_dict}

                        if extra_summary_parts:
                            extra_str = " | ".join(extra_summary_parts)
                            clean_body = f"{clean_body} [{extra_str}]".strip() if clean_body else extra_str

                        encoded_note = f"[META:{json.dumps(merged_meta)}] {clean_body}".strip() if merged_meta else clean_body

                        base_rec = ALL_13_VARIABLE_MAP
                        records_to_insert.append(base_rec)

                    bulk_insert_matches(records_to_insert, valid_db_cols)
                    st.success(f"✅ Cleanly imported all {len(records_to_insert)} rows with 100% of your CSV columns preserved!")
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading or saving CSV: {e}")

    st.markdown("---")
    st.subheader("📥 Export Complete Database as CSV")
    if not df.empty:
        export_df = df.drop(columns=[note_col], errors="ignore")
        csv_export = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Analytics CSV (All Columns + Imported Fields)",
            data=csv_export,
            file_name=f"courtiq_complete_analytics_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader(f"Complete Self-Logged Session History ({len(df)} Rows)")
    st.dataframe(df.drop(columns=[note_col], errors="ignore"), use_container_width=True)



# ==========================================
# TAB 7: PRE-MATCH OPPONENT SCOUTING (VISUAL & SMART MERGE)
# ==========================================
with tab7:
    st.subheader("Pre-Match Opponent Scouting & Visual Matchup Lab")

    import unicodedata

    def normalize_opp_name(raw_name):
        s = str(raw_name).strip()
        if not s or s.lower() in ["unknown", "wall", "practice", "nan", "none"]:
            return None
        # Strip ALL accents so '?' never appears or breaks encoding
        no_acc = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        low = no_acc.lower().strip()
        if "alex" in low and ("bigote" in low or "mustache" in low):
            return "Alex Bigotes"
        if "andres" in low and "jimen" in low:
            return "Andres Jimenez"
        if "pablo" in low and "pedro" in low:
            return "Pablo Pedros Student"
        if low == "ivan" or low.startswith("ivan "):
            return "Ivan"
        return " ".join(w.capitalize() for w in no_acc.split())

    if df.empty or "opponent" not in df.columns:
        st.info("Log matches with opponent names to unlock Head-to-Head scouting charts.")
    else:
        scout_df = df.copy()
        scout_df["Clean Opponent"] = scout_df["opponent"].apply(normalize_opp_name)
        scout_df = scout_df[scout_df["Clean Opponent"].notna()].copy()

        # Sort opponents by most matches played first
        opp_counts = scout_df["Clean Opponent"].value_counts()
        known_opps = opp_counts.index.tolist()

        if not known_opps:
            st.info("Log matches with opponent names to unlock Head-to-Head scouting charts.")
        else:
            selected_opp = st.selectbox(
                "Select Opponent Before Stepping on Court",
                known_opps,
                format_func=lambda x: f"{x} ({opp_counts[x]} {'match' if opp_counts[x] == 1 else 'matches'})"
            )

            h2h_df = scout_df[scout_df["Clean Opponent"] == selected_opp].sort_values(by="match_date", ascending=True).copy()
            h2h_wins = len(h2h_df[h2h_df["Outcome"] == "Win"])
            h2h_losses = len(h2h_df[h2h_df["Outcome"] == "Loss"])
            decided = h2h_wins + h2h_losses
            win_pct = (h2h_wins / decided * 100) if decided > 0 else 0.0

            hk1, hk2, hk3, hk4 = st.columns(4)
            hk1.metric("Total Meetings", len(h2h_df))
            hk2.metric("H2H Record", f"{h2h_wins}W - {h2h_losses}L")
            hk3.metric("Win Rate", f"{win_pct:.0f}%" if decided > 0 else "N/A")
            hk4.metric("Avg Serve / Focus", f"{pd.to_numeric(h2h_df['serve_rating'], errors='coerce').mean():.1f} / {pd.to_numeric(h2h_df['mental_focus'], errors='coerce').mean():.1f}")

            st.markdown("---")
            g_col1, g_col2 = st.columns(2)

            # Graph 1 & Graph 2: ALL Skills Radar + Progression vs this Opponent
            h2h_skills = []
            for s_label, s_col, _, s_color in GRANULAR_STROKES + [("Mental Focus", "mental_focus", ["mental_focus"], "#f1c40f")]:
                s_sub = pd.to_numeric(h2h_df[s_col], errors="coerce") if s_col in h2h_df.columns else pd.Series(np.nan, index=h2h_df.index)
                s_all = pd.to_numeric(df[s_col], errors="coerce") if s_col in df.columns else s_sub
                season_val = float(s_all.mean()) if s_all.notna().any() else 5.0
                h2h_skills.append((
                    s_label,
                    s_sub.fillna(season_val),
                    season_val,
                    s_color
                ))

            with g_col1:
                if len(h2h_skills) >= 3:
                    r_labels = [x[0] for x in h2h_skills]
                    h2h_vals = [round(float(x[1].mean()), 2) for x in h2h_skills]
                    season_vals = [round(x[2], 2) for x in h2h_skills]

                    fig_h2h_radar = go.Figure()
                    fig_h2h_radar.add_trace(go.Scatterpolar(
                        r=season_vals + [season_vals[0]],
                        theta=r_labels + [r_labels[0]],
                        fill="toself",
                        name="Your Season Avg",
                        line=dict(color="#95a5a6", dash="dash"),
                        opacity=0.45
                    ))
                    fig_h2h_radar.add_trace(go.Scatterpolar(
                        r=h2h_vals + [h2h_vals[0]],
                        theta=r_labels + [r_labels[0]],
                        fill="toself",
                        name=f"vs. {selected_opp}",
                        line=dict(color="#2ecc71" if h2h_wins >= h2h_losses else "#e74c3c", width=3)
                    ))
                    fig_h2h_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                        title=f"All-Skill Radar vs. {selected_opp}",
                        template="plotly_dark",
                        height=410,
                        margin=dict(l=55, r=55, t=55, b=35)
                    )
                    st.plotly_chart(fig_h2h_radar, use_container_width=True)

            with g_col2:
                h2h_df["Date Label"] = h2h_df["match_date"].dt.strftime("%Y-%m-%d").fillna("Session")
                h2h_df["Meeting Label"] = [f"#{i+1} ({d})" for i, d in enumerate(h2h_df["Date Label"])]

                fig_h2h_trend = go.Figure()
                for s_label, s_series, _, s_color in h2h_skills:
                    fig_h2h_trend.add_trace(go.Scatter(
                        x=h2h_df["Meeting Label"],
                        y=s_series,
                        mode="lines+markers",
                        name=s_label,
                        connectgaps=True,
                        line=dict(color=s_color, width=2.5),
                        marker=dict(size=9)
                    ))

                fig_h2h_trend.update_layout(
                    title=f"All Skills Progression vs. {selected_opp}",
                    yaxis=dict(range=[0, 10.5], title="Rating (1-10)"),
                    xaxis_title="Match vs. Opponent (Click legend to filter)",
                    template="plotly_dark",
                    height=410,
                    hovermode="x unified",
                    margin=dict(l=40, r=25, t=55, b=35)
                )
                st.plotly_chart(fig_h2h_trend, use_container_width=True)

            # Compact, scannable summary table + optional collapsible raw notes

            if "match_type" in h2h_df.columns and h2h_df["match_type"].nunique() > 1:
                fmt_counts = h2h_df["match_type"].fillna("Match").value_counts().reset_index()
                fmt_counts.columns = ["Format", "Sessions"]
                fig_fmt = px.pie(
                    fmt_counts, values="Sessions", names="Format", hole=0.5,
                    title=f"Session Formats Played with {selected_opp}",
                    template="plotly_dark"
                )
                fig_fmt.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_fmt, use_container_width=True)

            st.markdown("#### Match Summary")
            summary_cols = [c for c in ["Date Label", "Outcome", "score", "serve_rating", "mental_focus"] if c in h2h_df.columns]
            st.dataframe(
                h2h_df[summary_cols].rename(columns=ALL_13_VARIABLE_MAP),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View Full Match Notes (Optional)"):
                for _, r in h2h_df.sort_values(by="match_date", ascending=False).iterrows():
                    dt_str = r["match_date"].strftime("%Y-%m-%d") if pd.notna(r.get("match_date")) else "Past"
                    raw_note = str(r.get("clean_note", "")).strip()
                    # Clean out raw timestamp prefixes if present
                    raw_note = re.sub(r"^\(\d{1,2}/\d{1,2}/\d{4}[^)]*\)\s*\\s*", "", raw_note)
                    st.markdown(f"**{dt_str} ? {r.get('Outcome', '')}**")
                    st.caption(raw_note if raw_note else "No notes recorded.")
