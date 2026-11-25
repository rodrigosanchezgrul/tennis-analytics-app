import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import date
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# --- PAGE CONFIG ---
st.set_page_config(page_title="CourtIQ Pro | Tennis Analytics & ML", page_icon="🎾", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .coach-card {
        background-color: #161b22;
        border-left: 5px solid #2ecc71;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .loss-card {
        background-color: #161b22;
        border-left: 5px solid #e74c3c;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .wall-card {
        background-color: #161b22;
        border-left: 5px solid #3498db;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- SUPABASE CONNECTION ---
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- AUTHENTICATION (PIN LOCK) ---
def check_pin():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🎾 CourtIQ Pro Login")
        pin_input = st.text_input("Enter your access PIN", type="password")
        if st.button("Unlock"):
            if pin_input == st.secrets["APP_PIN"]:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect PIN. Try again.")
        st.stop()

check_pin()

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

def load_data():
    response = supabase.table("personal_matches").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return df

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

    # Categorize session into Match vs Training/Wall
    if "match_type" in df.columns:
        df["Category"] = np.where(
            df["match_type"].astype(str).str.contains("Wall|Drill|Coach|Practice", case=False, na=False),
            "Training & Wall",
            "Competitive Match"
        )
    else:
        df["Category"] = "Competitive Match"

    # Clean outcome label
    if "result" in df.columns:
        def map_outcome(val):
            s = str(val).lower().strip()
            if s.startswith("w"):
                return "Win"
            elif s.startswith("l"):
                return "Loss"
            return "Training / Wall"
        df["Outcome"] = df["result"].apply(map_outcome)
    else:
        df["Outcome"] = "Training / Wall"

    return df

def insert_match(record: dict, existing_cols: list):
    clean_record = {k: v for k, v in record.items() if k in existing_cols} if existing_cols else record
    supabase.table("personal_matches").insert(clean_record).execute()

# --- MAIN APP ---
st.title("🎾 CourtIQ Pro — Tennis Intelligence & ML Performance Lab")

df = load_data()
meta_cols = ["clean_note", "opp_style", "pressure", "physical_state", "fh_rating", "bh_rating", "ret_rating", "net_rating", "main_weapon", "main_leak", "Category", "Outcome"]
raw_cols = [c for c in df.columns if c not in meta_cols]
note_col = next((c for c in ["tactical_note", "tactical_notes", "notes", "note", "reflection", "comments"] if c in raw_cols), "tactical_note")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕸️ Skill Radar & Scatter Lab",
    "📊 Volume, Wins & Loss Reasons",
    "🤖 AI Coach & ML Models",
    "➕ Log Detailed Session",
    "📁 Full Database & AI Export"
])

# ==========================================
# TAB 1: PRO SKILL RADAR & SCATTERPLOT LAB
# ==========================================
with tab1:
    if df.empty:
        st.info("No sessions logged yet. Go to 'Log Detailed Session' to add your first match or wall session!")
    else:
        st.subheader("🕸️ Pro-Style 8-Axis Skill & Match Profile (Tennis Insights Style)")
        st.caption("Compare your overall Season Average against your Wins, Losses, and Wall/Practice sessions across every technical and mental dimension.")

        # Build the 8 radar axes using available columns (with smart fallbacks to core ratings for older rows)
        def get_axis_series(col_name, fallback_col):
            if col_name in df.columns and df[col_name].notna().any():
                s = pd.to_numeric(df[col_name], errors="coerce")
                fb = pd.to_numeric(df[fallback_col], errors="coerce") if fallback_col in df.columns else pd.Series(5.0, index=df.index)
                return s.fillna(fb).fillna(5.0)
            elif fallback_col in df.columns:
                return pd.to_numeric(df[fallback_col], errors="coerce").fillna(5.0)
            return pd.Series(5.0, index=df.index)

        radar_df = pd.DataFrame(index=df.index)
        radar_df["Serve Quality"] = get_axis_series("serve_rating", "serve_rating")
        radar_df["Return Quality"] = get_axis_series("ret_rating", "mental_focus")
        radar_df["Forehand Quality"] = get_axis_series("fh_rating", "serve_rating")
        radar_df["Backhand Quality"] = get_axis_series("bh_rating", "energy_level")
        radar_df["Net & Conversion"] = get_axis_series("net_rating", "serve_rating")
        radar_df["Baseline Battle (Focus)"] = get_axis_series("mental_focus", "mental_focus")
        radar_df["Physical Steal (Energy)"] = get_axis_series("energy_level", "energy_level")
        radar_df["Attack / Overall Index"] = radar_df.mean(axis=1)
        radar_df["Outcome"] = df["Outcome"]

        axes = [
            "Serve Quality", "Return Quality", "Forehand Quality", "Backhand Quality",
            "Net & Conversion", "Baseline Battle (Focus)", "Physical Steal (Energy)", "Attack / Overall Index"
        ]

        season_avg = [radar_df[a].mean() for a in axes]
        wins_df = radar_df[radar_df["Outcome"] == "Win"]
        losses_df = radar_df[radar_df["Outcome"] == "Loss"]
        wall_df = radar_df[radar_df["Outcome"] == "Training / Wall"]

        r_col1, r_col2 = st.columns([1.4, 1])

        with r_col1:
            fig_radar = go.Figure()

            # Trace 1: Season Average (Red, like 2025 Season Avg)
            fig_radar.add_trace(go.Scatterpolar(
                r=season_avg + [season_avg[0]],
                theta=axes + [axes[0]],
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.12)",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=8, color="#e74c3c"),
                name=f"Season Avg ({len(radar_df)} sessions)"
            ))

            # Trace 2: In Wins (Green, like When Playing Sinner)
            if not wins_df.empty:
                win_avg = [wins_df[a].mean() for a in axes]
                fig_radar.add_trace(go.Scatterpolar(
                    r=win_avg + [win_avg[0]],
                    theta=axes + [axes[0]],
                    fill="toself",
                    fillcolor="rgba(46, 204, 113, 0.18)",
                    line=dict(color="#2ecc71", width=3),
                    marker=dict(size=8, color="#2ecc71"),
                    name=f"In Wins ({len(wins_df)} matches)"
                ))

            # Trace 3: In Losses (Dark/Amber Benchmark)
            if not losses_df.empty:
                loss_avg = [losses_df[a].mean() for a in axes]
                fig_radar.add_trace(go.Scatterpolar(
                    r=loss_avg + [loss_avg[0]],
                    theta=axes + [axes[0]],
                    fill="none",
                    line=dict(color="#f39c12", width=2.5, dash="dot"),
                    marker=dict(size=7, color="#f39c12"),
                    name=f"In Losses ({len(losses_df)} matches)"
                ))

            # Trace 4: Wall / Training Sessions (Blue)
            if not wall_df.empty:
                wall_avg = [wall_df[a].mean() for a in axes]
                fig_radar.add_trace(go.Scatterpolar(
                    r=wall_avg + [wall_avg[0]],
                    theta=axes + [axes[0]],
                    fill="none",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=6, color="#3498db"),
                    name=f"Wall & Training ({len(wall_df)})"
                ))

            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#11161d",
                    radialaxis=dict(visible=True, range=[0, 10], tickvals=[2, 4, 6, 8, 10], gridcolor="#2c3540"),
                    angularaxis=dict(gridcolor="#2c3540", tickfont=dict(size=12))
                ),
                template="plotly_dark",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                margin=dict(l=60, r=60, t=60, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with r_col2:
            st.markdown("### 📋 Skill Scorecard Breakdown")
            scorecard = pd.DataFrame({
                "Skill Dimension": axes,
                "Season Avg": [round(x, 1) for x in season_avg],
                "Win Avg": [round(wins_df[a].mean(), 1) if not wins_df.empty else np.nan for a in axes],
                "Loss Avg": [round(losses_df[a].mean(), 1) if not losses_df.empty else np.nan for a in axes],
            })
            if not wins_df.empty and not losses_df.empty:
                scorecard["Win-Loss Gap"] = (scorecard["Win Avg"] - scorecard["Loss Avg"]).round(1)
            st.dataframe(scorecard, use_container_width=True, hide_index=True)

            # Identify top strength and biggest gap
            best_skill = scorecard.loc[scorecard["Season Avg"].idxmax(), "Skill Dimension"]
            best_val = scorecard["Season Avg"].max()
            st.success(f"🏆 **Highest Rated Skill:** **{best_skill}** ({best_val:.1f} / 10)")
            if "Win-Loss Gap" in scorecard.columns and scorecard["Win-Loss Gap"].notna().any():
                biggest_gap_row = scorecard.loc[scorecard["Win-Loss Gap"].idxmax()]
                st.warning(
                    f"⚡ **Biggest Win/Loss Separator:** **{biggest_gap_row['Skill Dimension']}** "
                    f"drops by **{biggest_gap_row['Win-Loss Gap']:.1f} pts** when you lose."
                )

        st.markdown("---")
        st.subheader("🔵 Multi-Factor Correlation Scatterplot")
        st.caption("Explore how any two skills interact across your matches and training sessions. Hover over any dot to read your match notes.")

        sc1, sc2, sc3 = st.columns(3)
        x_axis = sc1.selectbox("X-Axis Metric", axes, index=5)  # Default: Baseline Battle (Focus)
        y_axis = sc2.selectbox("Y-Axis Metric", axes, index=6)  # Default: Physical Steal (Energy)
        show_trend = sc3.checkbox("Show OLS Trendline", value=True)

        scatter_plot_df = radar_df.copy()
        scatter_plot_df["Date"] = df["match_date"].dt.strftime("%Y-%m-%d") if "match_date" in df.columns else ""
        scatter_plot_df["Opponent / Session"] = df["opponent"] if "opponent" in df.columns else ""
        scatter_plot_df["Score"] = df["score"] if "score" in df.columns else ""
        scatter_plot_df["Notes"] = df["clean_note"].fillna("") if "clean_note" in df.columns else ""

        # Add slight jitter so identical integer ratings don't completely overlap
        rng = np.random.default_rng(42)
        scatter_plot_df["x_jit"] = scatter_plot_df[x_axis] + rng.uniform(-0.12, 0.12, size=len(scatter_plot_df))
        scatter_plot_df["y_jit"] = scatter_plot_df[y_axis] + rng.uniform(-0.12, 0.12, size=len(scatter_plot_df))

        fig_scatter = px.scatter(
            scatter_plot_df,
            x="x_jit",
            y="y_jit",
            color="Outcome",
            size="Attack / Overall Index",
            hover_data={"x_jit": False, "y_jit": False, x_axis: ":.1f", y_axis: ":.1f", "Date": True, "Opponent / Session": True, "Score": True, "Notes": True},
            color_discrete_map={"Win": "#2ecc71", "Loss": "#e74c3c", "Training / Wall": "#3498db"},
            trendline="ols" if show_trend else None,
            labels={"x_jit": x_axis, "y_jit": y_axis},
            template="plotly_dark"
        )
        # Add quadrant reference lines at season averages
        fig_scatter.add_vline(x=scatter_plot_df[x_axis].mean(), line_dash="dash", line_color="gray", annotation_text="Avg")
        fig_scatter.add_hline(y=scatter_plot_df[y_axis].mean(), line_dash="dash", line_color="gray", annotation_text="Avg")
        fig_scatter.update_layout(height=450, xaxis=dict(range=[0.5, 10.5]), yaxis=dict(range=[0.5, 10.5]))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 2: VOLUME, WINS & LOSS REASONS
# ==========================================
with tab2:
    if df.empty:
        st.info("Log sessions first to see your training volume and loss diagnostics.")
    else:
        st.subheader("📊 Court Activity, Win/Loss Totals & Root-Cause Breakdown")

        total_all = len(df)
        total_matches = len(df[df["Outcome"].isin(["Win", "Loss"])])
        total_wins = len(df[df["Outcome"] == "Win"])
        total_losses = len(df[df["Outcome"] == "Loss"])
        total_wall = len(df[df["match_type"].astype(str).str.contains("Wall", case=False, na=False)]) if "match_type" in df.columns else 0
        total_drills = len(df[df["Outcome"] == "Training / Wall"]) - total_wall
        win_pct = (total_wins / total_matches * 100) if total_matches > 0 else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Court Days", total_all)
        k2.metric("Matches Played", total_matches)
        k3.metric("Total Wins", total_wins)
        k4.metric("Total Losses", total_losses)
        k5.metric("Win Rate", f"{win_pct:.0f}%")
        k6.metric("Wall / Training", f"{total_wall} Wall / {max(0, total_drills)} Drill")

        st.markdown("---")
        v1, v2 = st.columns(2)

        # Chart A: Training vs Match Volume Breakdown
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

        # Chart B: Monthly Activity & Win/Loss Trend
        with v2:
            st.subheader("Monthly Training & Match Consistency")
            if "match_date" in df.columns and df["match_date"].notna().any():
                monthly = df.copy()
                monthly["Month"] = monthly["match_date"].dt.to_period("M").astype(str)
                m_grp = monthly.groupby(["Month", "Outcome"]).size().reset_index(name="Sessions")
                fig_month = px.bar(
                    m_grp, x="Month", y="Sessions", color="Outcome", barmode="stack",
                    color_discrete_map={"Win": "#2ecc71", "Loss": "#e74c3c", "Training / Wall": "#3498db"},
                    template="plotly_dark"
                )
                fig_month.update_layout(height=340)
                st.plotly_chart(fig_month, use_container_width=True)

        st.markdown("---")
        l_col1, l_col2 = st.columns(2)

        # Chart C: Most Repeated Reasons for Losing (Tagged Leaks + Keyword Mining of Notes)
        with l_col1:
            st.subheader("🚨 Most Repeated Reasons for Losing")
            loss_rows = df[df["Outcome"] == "Loss"]
            if loss_rows.empty:
                st.success("Zero losses recorded yet!")
            else:
                reasons = []
                # 1. Pull explicit leak tags from losses
                if "main_leak" in loss_rows.columns:
                    for item in loss_rows["main_leak"].dropna():
                        if item != "None":
                            reasons.append(item)

                # 2. Keyword-mine historical tactical notes from losses so older matches are included too!
                if "clean_note" in loss_rows.columns:
                    keyword_map = {
                        "Backhand errors / depth": ["backhand", "bh", "revés", "reves"],
                        "Forehand unforced errors": ["forehand", "fh", "derecha"],
                        "Serve / Double faults": ["serve", "double fault", "saque", "doble falta"],
                        "Mental focus / Impatience": ["focus", "mental", "rush", "nerv", "concentra", "paciencia"],
                        "Footwork / Tired legs": ["footwork", "legs", "tired", "slow", "pies", "cansad", "movimiento"],
                        "Return of serve": ["return", "devolucion", "devolución"]
                    }
                    for note in loss_rows["clean_note"].dropna():
                        low = str(note).lower()
                        for label, kws in keyword_map.items():
                            if any(k in low for k in kws):
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
                    st.info("Your older loss notes didn't specify a cause. Use the 'Primary Leak' dropdown in Tab 4 when logging losses to track exact patterns.")

        # Chart D: Top Strengths & Winning Weapons
        with l_col2:
            st.subheader("💪 Your Best Strengths & Winning Weapons")
            strengths = []
            if "main_weapon" in df.columns:
                for w in df["main_weapon"].dropna():
                    strengths.append(w)

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
                st.info("Log sessions with the 'Primary Weapon' tag to chart your top strengths.")

# ==========================================
# TAB 3: AI COACH & ML MODELS
# ==========================================
with tab3:
    st.subheader("🤖 AI Coach Diagnostic & Machine Learning Lab")
    st.caption("Combines Random Forest, XGBoost, Logistic Regression, and Ridge Regression to explain WHY you win, WHY you lose, and what to train next.")

    ml_df = df[df["Outcome"].isin(["Win", "Loss"])].copy()

    if len(ml_df) < 6:
        st.warning(f"Currently found {len(ml_df)} completed matches. Log at least 6–8 matches with Win/Loss results to unlock the ML diagnostic engine.")
    else:
        ml_df["target_win"] = np.where(ml_df["Outcome"] == "Win", 1, 0)

        num_features = [c for c in ["serve_rating", "mental_focus", "energy_level", "fh_rating", "bh_rating", "ret_rating", "net_rating"] if c in ml_df.columns]
        X_num = ml_df[num_features].apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median()).fillna(5.0)

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
            ridge_reg = Ridge(alpha=1.0).fit(X_scaled, X_num.mean(axis=1))

            # --- AUTOMATED AI COACHING NARRATIVE ---
            win_means = X_num[y == 1].mean()
            loss_means = X_num[y == 0].mean()
            diffs = (win_means - loss_means).sort_values(ascending=False)

            biggest_win_driver = diffs.index[0]
            second_win_driver = diffs.index[1] if len(diffs) > 1 else diffs.index[0]
            weakest_in_losses = loss_means.idxmin()

            # Check wall training volume
            wall_count = len(df[df["Outcome"] == "Training / Wall"])
            match_count = len(ml_df)

            st.markdown("### 🧠 Automated AI Coach Report (Generated From Your Data)")
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
                ratio_msg = (
                    f"You have logged <b>{wall_count} training/wall sessions</b> vs. <b>{match_count} matches</b>. Increase wall reps focused on <b>{weakest_in_losses}</b> before your next match."
                    if wall_count < match_count
                    else f"Great <b>{wall_count}:{match_count}</b> training-to-match ratio! Dedicate your next wall session to high-rep consistency on <b>{weakest_in_losses}</b>."
                )
                st.markdown(f"""
                <div class="wall-card">
                    <h4>🧱 Wall & Match Training Plan</h4>
                    <p><b>#1 Priority Drill:</b> Target <b>{weakest_in_losses}</b> & <b>{biggest_win_driver}</b>.</p>
                    <p>{ratio_msg}</p>
                </div>
                """, unsafe_allow_html=True)

            # Model Accuracy Strip
            cv_folds = min(3, int(min(y.sum(), len(y) - y.sum())))
            if cv_folds >= 2:
                rf_acc = cross_val_score(rf_clf, X, y, cv=cv_folds).mean()
                xgb_acc = cross_val_score(xgb_clf, X, y, cv=cv_folds).mean()
                log_acc = cross_val_score(log_reg, X_scaled, y, cv=cv_folds).mean()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Random Forest Accuracy", f"{rf_acc*100:.0f}%")
                m2.metric("XGBoost Accuracy", f"{xgb_acc*100:.0f}%")
                m3.metric("Logistic Regression Accuracy", f"{log_acc*100:.0f}%")
                m4.metric("Ridge Regression R²", f"{ridge_reg.score(X_scaled, X_num.mean(axis=1)):.2f}")

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

            # Interactive What-If Simulator
            st.markdown("---")
            st.subheader("🎛️ Interactive 'What-If' Match Win Simulator")
            st.caption("Slide your target execution levels for your next match to see how all 3 classification models update your Win Probability:")
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
# TAB 4: LOG DETAILED SESSION
# ==========================================
with tab4:
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
        st.markdown("**Match Context & Opponent Profile**")
        ctx1, ctx2, ctx3 = st.columns(3)
        opp_style = ctx1.selectbox(
            "Opponent Playstyle / Drill Theme",
            ["All-Court", "Aggressive Baseliner", "Counterpuncher / Pusher", "Big Server", "Net Rusher / S&V", "Lefty / Unorthodox", "Solo Wall / Footwork"]
        )
        pressure = ctx2.selectbox(
            "Match Flow / Pressure Context",
            ["Competitive / Close Sets", "Tiebreak / 3rd Set Decider", "Dominant Control", "Played From Behind", "Routine Practice"]
        )
        physical_state = ctx3.selectbox(
            "Physical Readiness",
            ["Fresh & Explosive", "Normal", "Heavy Legs / Fatigued", "Sore / Tight"]
        )

        st.markdown("---")
        st.markdown("**8-Axis Skill Ratings (Powers Your Pro Radar Chart — 1 to 10)**")
        r1, r2, r3, r4 = st.columns(4)
        serve_rating = r1.slider("Serve Quality", 1, 10, 6)
        ret_rating = r2.slider("Return Quality", 1, 10, 6)
        fh_rating = r3.slider("Forehand Quality", 1, 10, 6)
        bh_rating = r4.slider("Backhand Quality", 1, 10, 6)

        r5, r6, r7 = st.columns(3)
        net_rating = r5.slider("Net & Conversion", 1, 10, 6)
        mental_focus = r6.slider("Baseline Battle / Focus", 1, 10, 6)
        energy_level = r7.slider("Physical Steal / Energy", 1, 10, 6)

        st.markdown("---")
        st.markdown("**Tactical Root-Cause & Descriptive Notes**")
        t1, t2 = st.columns(2)
        main_weapon = t1.selectbox(
            "Primary Weapon / Strength Today",
            ["Forehand", "Backhand", "First Serve", "Return", "Footwork / Defense", "Net Play", "Mental Composure"]
        )
        main_leak = t2.selectbox(
            "Primary Reason for Losing / Main Leak",
            ["Unforced errors on Forehand", "Short / weak Backhand", "Low 1st serve % / Double faults", "Slow split-step / Footwork", "Rushing points / Impatience", "Return depth", "None"]
        )

        n1, n2 = st.columns(2)
        what_worked = n1.text_area(
            "What worked well & tactical patterns that won points?",
            placeholder="e.g., Heavy crosscourt forehand pushed opponent wide, opening up the down-the-line winner."
        )
        what_to_adjust = n2.text_area(
            "Where did you lose points & what is the #1 drill/focus next time?",
            placeholder="e.g., Struggled returning kick serves to the backhand; need to drill backhand depth on the wall."
        )

        submitted = st.form_submit_button("Save Detailed Session", use_container_width=True)
        if submitted:
            meta_payload = {
                "opp_style": opp_style,
                "pressure": pressure,
                "physical_state": physical_state,
                "fh_rating": fh_rating,
                "bh_rating": bh_rating,
                "ret_rating": ret_rating,
                "net_rating": net_rating,
                "main_weapon": main_weapon,
                "main_leak": main_leak
            }
            narrative_parts = []
            if what_worked.strip():
                narrative_parts.append(f"Worked: {what_worked.strip()}")
            if what_to_adjust.strip():
                narrative_parts.append(f"Adjust: {what_to_adjust.strip()}")
            human_summary = f"[{opp_style} | {pressure} | Weapon: {main_weapon} | Leak: {main_leak}] " + " // ".join(narrative_parts)

            encoded_note = f"[META:{json.dumps(meta_payload)}] {human_summary}".strip()

            new_record = {
                "match_date": str(session_date),
                "match_type": match_type,
                "surface": surface,
                "opponent": opponent if opponent else "Wall",
                "result": result,
                "score": score,
                "serve_rating": serve_rating,
                "mental_focus": mental_focus,
                "energy_level": energy_level,
                note_col: encoded_note
            }
            insert_match(new_record, raw_cols)
            st.success("Session saved! Your Skill Radar, Scatterplot, and ML Coach have been updated.")
            st.rerun()

# ==========================================
# TAB 5: FULL DATABASE & AI EXPORT
# ==========================================
with tab5:
    st.subheader("Export Enriched Dataset for NotebookLM / Gemini")
    if not df.empty:
        export_df = df.drop(columns=[note_col], errors="ignore")
        csv_export = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Analytics CSV for External AI",
            data=csv_export,
            file_name=f"courtiq_pro_analytics_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("Log sessions first to enable CSV export.")

    st.markdown("---")
    st.subheader("Complete Session Log")
    st.dataframe(df.drop(columns=[note_col], errors="ignore"), use_container_width=True)
