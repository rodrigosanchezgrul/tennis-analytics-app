from datetime import date, datetime
import hmac
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Page Configuration & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CourtIQ | Personal Tennis Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(34, 197, 94, 0.06);
            border: 1px solid rgba(34, 197, 94, 0.22);
            border-radius: 12px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label {
            font-weight: 600;
            font-size: 0.85rem;
        }
        .note-card {
            border-left: 4px solid #22c55e;
            background-color: rgba(148, 163, 184, 0.08);
            border-radius: 0 10px 10px 0;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        .note-card-loss {
            border-left: 4px solid #ef4444;
        }
        .tag-pill {
            display: inline-block;
            background-color: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.78rem;
            margin-right: 6px;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATCH_TYPES = ["Tournament", "League", "Practice Match", "Drills"]
COURT_SURFACES = ["Hard", "Clay", "Grass", "Indoor"]
RESULTS = ["Win", "Loss"]
TACTICAL_TAGS = [
    "Forehand Winner",
    "Backhand Errors",
    "Net Play",
    "Return Depth",
    "Footwork",
    "Second Serve",
]
SURFACE_COLORS = {
    "Hard": "#3b82f6",
    "Clay": "#f97316",
    "Grass": "#22c55e",
    "Indoor": "#a855f7",
}


# ---------------------------------------------------------------------------
# Authentication Gate (PIN via st.secrets["APP_PIN"])
# ---------------------------------------------------------------------------
def require_pin_auth() -> None:
    """Enforces PIN authentication before rendering any match data or forms."""
    if st.session_state.get("authenticated", False):
        return

    expected_pin = str(st.secrets.get("APP_PIN", "")).strip()
    if not expected_pin:
        st.error(
            "Missing `APP_PIN` in Streamlit secrets. Add `APP_PIN`, `SUPABASE_URL`, "
            "and `SUPABASE_KEY` in your Streamlit Cloud App Settings -> Secrets."
        )
        st.stop()

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown("## 🎾 CourtIQ Tennis Log")
        st.caption("Private Personal Match Tracker & Performance Analytics")
        with st.form("pin_gate_form"):
            entered_pin = st.text_input(
                "Enter Access PIN",
                type="password",
                placeholder="••••",
            )
            submitted = st.form_submit_button(
                "Unlock Dashboard",
                type="primary",
                use_container_width=True,
            )
            if submitted:
                if hmac.compare_digest(entered_pin.strip(), expected_pin):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect PIN. Please try again.")

    st.stop()


# ---------------------------------------------------------------------------
# Supabase Client & Cached Data Access Layer
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = str(st.secrets["SUPABASE_URL"]).strip()
    key = str(st.secrets["SUPABASE_KEY"]).strip()
    return create_client(url, key)


@st.cache_data(ttl=600)
def fetch_matches_raw() -> list[dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table("personal_matches")
        .select("*")
        .order("match_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def parse_sets(scoreline: Any) -> list[str]:
    """Extracts set score tokens such as '6-4', '7-6(5)', or '[10-7]'."""
    if scoreline is None or (isinstance(scoreline, float) and pd.isna(scoreline)):
        return []
    text = str(scoreline).strip()
    if not text:
        return []
    return re.findall(r"\[?(\d+-\d+(?:\(\d+\))?)\]?", text)


def load_matches_df() -> pd.DataFrame:
    """Loads matches from cache and enriches with analytics helper columns."""
    rows = fetch_matches_raw()
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "created_at",
                "match_date",
                "opponent_name",
                "match_type",
                "court_surface",
                "result",
                "scoreline",
                "serve_rating",
                "mental_focus",
                "physical_energy",
                "first_serve_pct",
                "double_faults",
                "tags",
                "tactical_note",
                "is_win",
                "sets_list",
                "is_deciding_set",
            ]
        )

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["opponent_name"] = df["opponent_name"].fillna("Unknown").astype(str).str.strip()
    df["match_type"] = df["match_type"].fillna("Practice Match").astype(str)
    df["court_surface"] = df["court_surface"].fillna("Hard").astype(str)
    df["result"] = df["result"].fillna("Loss").astype(str)
    df["scoreline"] = df["scoreline"].fillna("").astype(str)
    df["tactical_note"] = df["tactical_note"].fillna("").astype(str)

    for rating_col in ["serve_rating", "mental_focus", "physical_energy"]:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(7).astype(int)

    df["first_serve_pct"] = pd.to_numeric(df["first_serve_pct"], errors="coerce")
    df["double_faults"] = pd.to_numeric(df["double_faults"], errors="coerce")
    df["tags"] = df["tags"].apply(lambda val: val if isinstance(val, list) else [])
    df["is_win"] = (df["result"] == "Win").astype(int)
    df["sets_list"] = df["scoreline"].apply(parse_sets)
    df["is_deciding_set"] = df["sets_list"].apply(lambda sets: len(sets) >= 3)

    df = df.sort_values(by=["match_date", "created_at"], ascending=[False, False]).reset_index(drop=True)
    return df


def insert_match(record: dict[str, Any]) -> None:
    client = get_supabase_client()
    client.table("personal_matches").insert(record).execute()
    st.cache_data.clear()


def bulk_insert_matches(records: list[dict[str, Any]], batch_size: int = 200) -> int:
    client = get_supabase_client()
    inserted = 0
    for idx in range(0, len(records), batch_size):
        batch = records[idx : idx + batch_size]
        client.table("personal_matches").insert(batch).execute()
        inserted += len(batch)
    st.cache_data.clear()
    return inserted


def update_match(match_id: str, updates: dict[str, Any]) -> None:
    client = get_supabase_client()
    client.table("personal_matches").update(updates).eq("id", match_id).execute()
    st.cache_data.clear()


def delete_match(match_id: str) -> None:
    client = get_supabase_client()
    client.table("personal_matches").delete().eq("id", match_id).execute()
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Analytics Helper Functions
# ---------------------------------------------------------------------------
def compute_current_streak(df: pd.DataFrame) -> tuple[str, str]:
    """Returns (streak_label, delta_description) from chronological matches."""
    if df.empty:
        return "—", "No matches yet"
    ordered = df.sort_values(by=["match_date", "created_at"], ascending=[True, True])
    results = ordered["result"].tolist()
    latest = results[-1]
    prefix = "W" if latest == "Win" else "L"
    count = 0
    for res in reversed(results):
        if res == latest:
            count += 1
        else:
            break
    last_opp = ordered.iloc[-1]["opponent_name"]
    return f"{prefix}{count}", f"Last vs {last_opp}"


def compute_best_surface(df: pd.DataFrame) -> tuple[str, str]:
    """Returns (surface_win_pct_label, record_subtext)."""
    if df.empty:
        return "—", "No surface data"
    grouped = (
        df.groupby("court_surface", as_index=False)
        .agg(matches=("is_win", "count"), wins=("is_win", "sum"))
    )
    grouped["win_pct"] = (grouped["wins"] / grouped["matches"]) * 100.0
    grouped = grouped.sort_values(by=["win_pct", "matches"], ascending=[False, False])
    top = grouped.iloc[0]
    losses = int(top["matches"] - top["wins"])
    return (
        f"{top['court_surface']} ({top['win_pct']:.0f}%)",
        f"{int(top['wins'])}W - {losses}L",
    )


def apply_clean_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.16)")
    return fig


def radio_or_pills(label: str, options: list[str], default_index: int = 0, key: str | None = None) -> str:
    """Uses st.pills when available (Streamlit 1.40+) with clean fallback to horizontal radio."""
    if hasattr(st, "pills"):
        selected = st.pills(
            label,
            options=options,
            default=options[default_index],
            selection_mode="single",
            key=key,
        )
        return selected if selected in options else options[default_index]
    return st.radio(label, options=options, index=default_index, horizontal=True, key=key)


# ---------------------------------------------------------------------------
# Tab 1: 30-Second Mobile Match Check-In
# ---------------------------------------------------------------------------
def render_log_match_tab(df: pd.DataFrame) -> None:
    st.subheader("30-Second Post-Match Check-In")
    st.caption("Log your match right off the court. Detailed stats are optional.")

    existing_opponents = sorted(df["opponent_name"].dropna().unique().tolist()) if not df.empty else []
    opponent_choices = ["➕ New Opponent..."] + existing_opponents

    with st.form("log_new_match_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 1.4])
        with col1:
            match_date = st.date_input("Match Date", value=date.today(), max_value=date.today())
        with col2:
            selected_opponent = st.selectbox(
                "Opponent (Past Opponents)",
                options=opponent_choices,
                index=0,
            )

        new_opponent_text = st.text_input(
            "New Opponent Name (if adding someone new)",
            placeholder="e.g., Alex Rivera (leave blank if selected from dropdown above)",
        )

        type_col, surf_col = st.columns(2)
        with type_col:
            match_type = radio_or_pills("Match Type", MATCH_TYPES, default_index=0, key="log_match_type")
        with surf_col:
            court_surface = radio_or_pills("Court Surface", COURT_SURFACES, default_index=0, key="log_surface")

        res_col, score_col = st.columns([1, 1.4])
        with res_col:
            result = radio_or_pills("Result", RESULTS, default_index=0, key="log_result")
        with score_col:
            scoreline = st.text_input(
                "Scoreline",
                placeholder="e.g., 6-4, 3-6, 10-7",
            )

        st.markdown("**Quick Performance Ratings (1–10)**")
        s_col, m_col, p_col = st.columns(3)
        with s_col:
            serve_rating = st.slider("🎾 Serve Rating", min_value=1, max_value=10, value=7)
        with m_col:
            mental_focus = st.slider("🧠 Mental Focus", min_value=1, max_value=10, value=7)
        with p_col:
            physical_energy = st.slider("⚡ Physical Energy", min_value=1, max_value=10, value=7)

        with st.expander("➕ Add Detailed Stats & Tactical Notes", expanded=False):
            stat_c1, stat_c2 = st.columns(2)
            with stat_c1:
                first_serve_pct = st.number_input(
                    "1st Serve % (optional)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    step=1.0,
                    format="%.1f",
                    placeholder="e.g., 64.0",
                )
            with stat_c2:
                double_faults = st.number_input(
                    "Double Faults (optional)",
                    min_value=0,
                    max_value=50,
                    value=None,
                    step=1,
                    placeholder="e.g., 3",
                )

            selected_tags = st.multiselect(
                "Main Weapon / Main Leak",
                options=TACTICAL_TAGS,
                placeholder="Select key patterns from today's match",
            )
            tactical_note = st.text_area(
                "Tactical Note",
                placeholder="What worked? What should I adjust next time?",
                height=95,
            )

        submitted = st.form_submit_button(
            "💾 Log Match to Cloud",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            typed_name = new_opponent_text.strip()
            if typed_name:
                case_map = {opp.lower(): opp for opp in existing_opponents}
                final_opponent = case_map.get(typed_name.lower(), typed_name)
            elif selected_opponent != "➕ New Opponent...":
                final_opponent = selected_opponent.strip()
            else:
                st.error("Please select an existing opponent or enter a new opponent's name.")
                return

            payload = {
                "match_date": match_date.isoformat(),
                "opponent_name": final_opponent,
                "match_type": match_type,
                "court_surface": court_surface,
                "result": result,
                "scoreline": scoreline.strip() if scoreline.strip() else None,
                "serve_rating": int(serve_rating),
                "mental_focus": int(mental_focus),
                "physical_energy": int(physical_energy),
                "first_serve_pct": float(first_serve_pct) if first_serve_pct is not None else None,
                "double_faults": int(double_faults) if double_faults is not None else None,
                "tags": list(selected_tags),
                "tactical_note": tactical_note.strip() if tactical_note.strip() else None,
            }

            try:
                insert_match(payload)
                st.toast(f"Saved {result} vs {final_opponent}!", icon="🎾")
                st.success(f"Logged **{result}** vs **{final_opponent}** ({court_surface} • {match_type}).")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save match to Supabase: {exc}")


# ---------------------------------------------------------------------------
# Tab 2: Performance Analytics & Trends (Plotly)
# ---------------------------------------------------------------------------
def render_analytics_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No matches logged yet. Add your first match in **🎾 Log New Match** or import a CSV in **📋 Match Log & Importer**.")
        return

    total_matches = len(df)
    total_wins = int(df["is_win"].sum())
    total_losses = total_matches - total_wins
    overall_win_rate = (total_wins / total_matches) * 100.0

    streak_label, streak_sub = compute_current_streak(df)
    best_surf_label, best_surf_sub = compute_best_surface(df)

    current_year = date.today().year
    ytd_df = df[pd.to_datetime(df["match_date"]).dt.year == current_year]
    ytd_matches = len(ytd_df)
    ytd_wins = int(ytd_df["is_win"].sum()) if ytd_matches else 0
    ytd_losses = ytd_matches - ytd_wins

    deciding_df = df[df["is_deciding_set"]]
    deciding_total = len(deciding_df)
    if deciding_total > 0:
        deciding_wins = int(deciding_df["is_win"].sum())
        deciding_losses = deciding_total - deciding_wins
        deciding_pct_str = f"{(deciding_wins / deciding_total) * 100.0:.1f}%"
        deciding_sub = f"{deciding_wins}W - {deciding_losses}L in 3rd/TB"
    else:
        deciding_pct_str = "—"
        deciding_sub = "0 deciding sets"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Overall Win Rate", f"{overall_win_rate:.1f}%", f"{total_wins}W - {total_losses}L")
    k2.metric("Current Streak", streak_label, streak_sub)
    k3.metric("Best Surface", best_surf_label, best_surf_sub)
    k4.metric(f"{current_year} YTD Matches", f"{ytd_matches}", f"{ytd_wins}W - {ytd_losses}L")
    k5.metric("Deciding Set / Super-TB", deciding_pct_str, deciding_sub)

    st.divider()

    st.markdown("### 📈 10-Match Rolling Form & Readiness Curve")
    chrono = df.sort_values(by=["match_date", "created_at"], ascending=[True, True]).reset_index(drop=True)
    chrono["match_num"] = chrono.index + 1
    chrono["rolling_win_pct"] = chrono["is_win"].rolling(window=10, min_periods=1).mean() * 100.0
    chrono["rolling_mental"] = chrono["mental_focus"].rolling(window=10, min_periods=1).mean()
    chrono["rolling_energy"] = chrono["physical_energy"].rolling(window=10, min_periods=1).mean()
    chrono["hover_label"] = (
        "Match #"
        + chrono["match_num"].astype(str)
        + " ("
        + chrono["match_date"].astype(str)
        + ")<br>vs "
        + chrono["opponent_name"]
        + " — "
        + chrono["result"]
        + " "
        + chrono["scoreline"]
    )

    fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rolling.add_trace(
        go.Scatter(
            x=chrono["match_num"],
            y=chrono["rolling_win_pct"],
            name="10-Match Rolling Win %",
            mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=6),
            text=chrono["hover_label"],
            hovertemplate="%{text}<br><b>Rolling Win %:</b> %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_rolling.add_trace(
        go.Scatter(
            x=chrono["match_num"],
            y=chrono["rolling_mental"],
            name="Rolling Mental Focus (1–10)",
            mode="lines",
            line=dict(color="#3b82f6", width=2.2, dash="dot"),
            hovertemplate="<b>Rolling Mental:</b> %{y:.2f}/10<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_rolling.add_trace(
        go.Scatter(
            x=chrono["match_num"],
            y=chrono["rolling_energy"],
            name="Rolling Physical Energy (1–10)",
            mode="lines",
            line=dict(color="#f59e0b", width=2.2, dash="dash"),
            hovertemplate="<b>Rolling Energy:</b> %{y:.2f}/10<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_rolling.update_xaxes(title_text="Chronological Match Number")
    fig_rolling.update_yaxes(title_text="Rolling Win %", range=[0, 105], secondary_y=False)
    fig_rolling.update_yaxes(
        title_text="Mental / Physical Rating (1–10)",
        range=[1, 10.5],
        showgrid=False,
        secondary_y=True,
    )
    fig_rolling = apply_clean_layout(fig_rolling, height=390)
    st.plotly_chart(fig_rolling, use_container_width=True)

    st.markdown("### 🏟️ Surface & Match Type Breakdown")
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        surf_order = [s for s in COURT_SURFACES if s in df["court_surface"].unique()]
        surf_agg = (
            df.groupby("court_surface", as_index=False)
            .agg(matches=("is_win", "count"), wins=("is_win", "sum"))
        )
        surf_agg["win_pct"] = (surf_agg["wins"] / surf_agg["matches"]) * 100.0
        surf_agg["court_surface"] = pd.Categorical(surf_agg["court_surface"], categories=surf_order, ordered=True)
        surf_agg = surf_agg.sort_values("court_surface")

        fig_surf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_surf.add_trace(
            go.Bar(
                x=surf_agg["court_surface"],
                y=surf_agg["win_pct"],
                name="Win %",
                marker_color="#22c55e",
                text=surf_agg["win_pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="outside",
                offsetgroup=0,
            ),
            secondary_y=False,
        )
        fig_surf.add_trace(
            go.Bar(
                x=surf_agg["court_surface"],
                y=surf_agg["matches"],
                name="Matches Played",
                marker_color="#3b82f6",
                text=surf_agg["matches"],
                textposition="outside",
                offsetgroup=1,
            ),
            secondary_y=True,
        )
        fig_surf.update_layout(title="Win % vs. Volume by Court Surface", barmode="group")
        fig_surf.update_yaxes(title_text="Win %", range=[0, 115], secondary_y=False)
        fig_surf.update_yaxes(
            title_text="Matches Played",
            range=[0, max(surf_agg["matches"].max() * 1.25, 4)],
            showgrid=False,
            secondary_y=True,
        )
        fig_surf = apply_clean_layout(fig_surf, height=360)
        st.plotly_chart(fig_surf, use_container_width=True)

    with b_col2:
        type_order = [t for t in MATCH_TYPES if t in df["match_type"].unique()]
        type_agg = (
            df.groupby("match_type", as_index=False)
            .agg(matches=("is_win", "count"), wins=("is_win", "sum"))
        )
        type_agg["win_pct"] = (type_agg["wins"] / type_agg["matches"]) * 100.0
        type_agg["match_type"] = pd.Categorical(type_agg["match_type"], categories=type_order, ordered=True)
        type_agg = type_agg.sort_values("match_type")

        fig_type = make_subplots(specs=[[{"secondary_y": True}]])
        fig_type.add_trace(
            go.Bar(
                x=type_agg["match_type"],
                y=type_agg["win_pct"],
                name="Win %",
                marker_color="#10b981",
                text=type_agg["win_pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="outside",
                offsetgroup=0,
            ),
            secondary_y=False,
        )
        fig_type.add_trace(
            go.Bar(
                x=type_agg["match_type"],
                y=type_agg["matches"],
                name="Matches Played",
                marker_color="#8b5cf6",
                text=type_agg["matches"],
                textposition="outside",
                offsetgroup=1,
            ),
            secondary_y=True,
        )
        fig_type.update_layout(title="Win % vs. Volume by Match Format", barmode="group")
        fig_type.update_yaxes(title_text="Win %", range=[0, 115], secondary_y=False)
        fig_type.update_yaxes(
            title_text="Matches Played",
            range=[0, max(type_agg["matches"].max() * 1.25, 4)],
            showgrid=False,
            secondary_y=True,
        )
        fig_type = apply_clean_layout(fig_type, height=360)
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("### 🔬 Win vs. Loss Diagnostics")
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        metrics_to_compare = [
            ("serve_rating", "Serve Rating (1–10)"),
            ("mental_focus", "Mental Focus (1–10)"),
            ("physical_energy", "Physical Energy (1–10)"),
            ("double_faults", "Avg Double Faults"),
        ]
        win_means = []
        loss_means = []
        labels = []
        wins_df = df[df["result"] == "Win"]
        losses_df = df[df["result"] == "Loss"]

        for col_name, display_label in metrics_to_compare:
            w_val = wins_df[col_name].dropna().mean() if not wins_df.empty else 0.0
            l_val = losses_df[col_name].dropna().mean() if not losses_df.empty else 0.0
            w_val = 0.0 if pd.isna(w_val) else float(w_val)
            l_val = 0.0 if pd.isna(l_val) else float(l_val)
            labels.append(display_label)
            win_means.append(w_val)
            loss_means.append(l_val)

        fig_diag = go.Figure()
        fig_diag.add_trace(
            go.Bar(
                x=labels,
                y=win_means,
                name="Wins",
                marker_color="#22c55e",
                text=[f"{v:.1f}" for v in win_means],
                textposition="outside",
            )
        )
        fig_diag.add_trace(
            go.Bar(
                x=labels,
                y=loss_means,
                name="Losses",
                marker_color="#ef4444",
                text=[f"{v:.1f}" for v in loss_means],
                textposition="outside",
            )
        )
        max_diag = max(win_means + loss_means + [10.0])
        fig_diag.update_layout(
            title="Ratings & Double Faults: Wins vs. Losses",
            barmode="group",
            yaxis=dict(range=[0, max_diag * 1.2]),
        )
        fig_diag = apply_clean_layout(fig_diag, height=370)
        st.plotly_chart(fig_diag, use_container_width=True)

        w_1st = wins_df["first_serve_pct"].dropna().mean() if not wins_df.empty else float("nan")
        l_1st = losses_df["first_serve_pct"].dropna().mean() if not losses_df.empty else float("nan")
        s1_c1, s1_c2 = st.columns(2)
        s1_c1.metric(
            "Avg 1st Serve % in Wins",
            f"{w_1st:.1f}%" if not pd.isna(w_1st) else "—",
            f"{int(wins_df['first_serve_pct'].notna().sum())} tracked wins",
        )
        s1_c2.metric(
            "Avg 1st Serve % in Losses",
            f"{l_1st:.1f}%" if not pd.isna(l_1st) else "—",
            f"{int(losses_df['first_serve_pct'].notna().sum())} tracked losses",
        )

    with diag_c2:
        exploded = df[["result", "tags"]].explode("tags")
        exploded = exploded[exploded["tags"].notna() & (exploded["tags"] != "")]

        if exploded.empty:
            st.info("No Weapon/Leak tags logged yet. Use the optional expander when logging matches to track key patterns.")
        else:
            tag_counts = (
                exploded.groupby(["tags", "result"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["Win", "Loss"], fill_value=0)
            )
            tag_counts["Total"] = tag_counts["Win"] + tag_counts["Loss"]
            tag_counts = tag_counts.sort_values("Total", ascending=True)

            fig_tags = go.Figure()
            fig_tags.add_trace(
                go.Bar(
                    y=tag_counts.index,
                    x=tag_counts["Win"],
                    name="Wins",
                    orientation="h",
                    marker_color="#22c55e",
                    text=tag_counts["Win"],
                    textposition="auto",
                )
            )
            fig_tags.add_trace(
                go.Bar(
                    y=tag_counts.index,
                    x=tag_counts["Loss"],
                    name="Losses",
                    orientation="h",
                    marker_color="#ef4444",
                    text=tag_counts["Loss"],
                    textposition="auto",
                )
            )
            fig_tags.update_layout(
                title="Most Frequent Weapon / Leak Tags (Wins vs. Losses)",
                barmode="group",
                xaxis_title="Match Count",
            )
            fig_tags = apply_clean_layout(fig_tags, height=455)
            st.plotly_chart(fig_tags, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Opponent Head-to-Head Scouting
# ---------------------------------------------------------------------------
def render_scouting_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Log at least one match to unlock Head-to-Head Opponent Scouting.")
        return

    opponents = sorted(df["opponent_name"].unique().tolist())
    selected_opp = st.selectbox("Select Opponent to Scout", options=opponents, index=0)

    opp_df = df[df["opponent_name"] == selected_opp].copy()
    opp_df = opp_df.sort_values(by=["match_date", "created_at"], ascending=[False, False]).reset_index(drop=True)

    wins = int(opp_df["is_win"].sum())
    losses = len(opp_df) - wins
    win_pct = (wins / len(opp_df)) * 100.0
    last_played = str(opp_df.iloc[0]["match_date"])
    avg_serve = opp_df["serve_rating"].mean()
    avg_mental = opp_df["mental_focus"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"H2H vs {selected_opp}", f"{wins}W - {losses}L", f"{win_pct:.1f}% Win Rate")
    m2.metric("Total Meetings", f"{len(opp_df)}", f"Last: {last_played}")
    m3.metric("Avg Serve vs Opponent", f"{avg_serve:.1f} / 10")
    m4.metric("Avg Mental Focus vs Opponent", f"{avg_mental:.1f} / 10")

    st.divider()

    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        st.markdown("#### 🏟️ Surface Split")
        surf_split = (
            opp_df.groupby("court_surface", as_index=False)
            .agg(Matches=("is_win", "count"), Wins=("is_win", "sum"))
        )
        surf_split["Losses"] = surf_split["Matches"] - surf_split["Wins"]
        surf_split["Win %"] = ((surf_split["Wins"] / surf_split["Matches"]) * 100.0).round(1)
        surf_split = surf_split[["court_surface", "Wins", "Losses", "Win %"]].rename(
            columns={"court_surface": "Surface"}
        )
        st.dataframe(surf_split, use_container_width=True, hide_index=True)

    with right_col:
        st.markdown("#### 🎾 Set-by-Set Match History")
        history_rows = []
        for _, row in opp_df.iterrows():
            sets = row["sets_list"]
            history_rows.append(
                {
                    "Date": str(row["match_date"]),
                    "Type": row["match_type"],
                    "Surface": row["court_surface"],
                    "Result": row["result"],
                    "Scoreline": row["scoreline"] or "—",
                    "Set 1": sets[0] if len(sets) >= 1 else "—",
                    "Set 2": sets[1] if len(sets) >= 2 else "—",
                    "Set 3 / TB": " / ".join(sets[2:]) if len(sets) >= 3 else "—",
                }
            )
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

    st.markdown(f"#### 📝 Chronological Tactical Scouting Notes vs. {selected_opp}")
    noted_matches = opp_df[
        (opp_df["tactical_note"].str.len() > 0) | (opp_df["tags"].apply(len) > 0)
    ]

    if noted_matches.empty:
        st.caption("No tactical notes or tags recorded against this opponent yet.")
    else:
        for _, row in noted_matches.iterrows():
            card_cls = "note-card" if row["result"] == "Win" else "note-card note-card-loss"
            tags_html = "".join(
                f'<span class="tag-pill">{tag}</span>' for tag in row["tags"]
            )
            note_body = row["tactical_note"] if row["tactical_note"] else "<em>No written note — tags logged.</em>"
            score_display = f" ({row['scoreline']})" if row["scoreline"] else ""
            header_text = (
                f"<strong>{row['match_date']}</strong> • "
                f"<strong>{row['result']}{score_display}</strong> • "
                f"{row['court_surface']} ({row['match_type']})"
            )
            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div>{header_text}</div>
                    <div style="margin-top: 6px;">{note_body}</div>
                    <div style="margin-top: 6px;">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 4: Match Log Management & Google Sheets CSV Importer
# ---------------------------------------------------------------------------
def normalize_result_value(raw_val: Any) -> str:
    val = str(raw_val).strip().lower()
    if val in {"w", "win", "won", "1", "true", "victory"}:
        return "Win"
    return "Loss"


def normalize_surface_value(raw_val: Any) -> str:
    val = str(raw_val).strip().lower()
    if "clay" in val or "dirt" in val:
        return "Clay"
    if "grass" in val:
        return "Grass"
    if "indoor" in val or "carpet" in val:
        return "Indoor"
    return "Hard"


def normalize_match_type_value(raw_val: Any) -> str:
    val = str(raw_val).strip().lower()
    if "tour" in val or "open" in val or "championship" in val:
        return "Tournament"
    if "league" in val or "usta" in val or "team" in val:
        return "League"
    if "drill" in val or "clinic" in val or "lesson" in val:
        return "Drills"
    return "Practice Match"


def render_log_and_importer_tab(df: pd.DataFrame) -> None:
    st.subheader("📋 Searchable Match Log")

    if df.empty:
        st.info("Your match log is currently empty. Log a match in Tab 1 or import your Google Sheets CSV below.")
    else:
        f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1])
        with f1:
            search_query = st.text_input(
                "Search Opponent, Scoreline, Notes, or Tags",
                placeholder="Type to filter...",
            )
        with f2:
            surf_filter = st.multiselect("Surface", options=COURT_SURFACES, default=[])
        with f3:
            type_filter = st.multiselect("Match Type", options=MATCH_TYPES, default=[])
        with f4:
            res_filter = st.multiselect("Result", options=RESULTS, default=[])

        f5, f6 = st.columns([1.4, 1.6])
        with f5:
            opp_filter = st.multiselect(
                "Opponent",
                options=sorted(df["opponent_name"].unique().tolist()),
                default=[],
            )
        with f6:
            min_d = df["match_date"].min()
            max_d = df["match_date"].max()
            date_range = st.date_input(
                "Date Range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )

        filtered = df.copy()
        if surf_filter:
            filtered = filtered[filtered["court_surface"].isin(surf_filter)]
        if type_filter:
            filtered = filtered[filtered["match_type"].isin(type_filter)]
        if res_filter:
            filtered = filtered[filtered["result"].isin(res_filter)]
        if opp_filter:
            filtered = filtered[filtered["opponent_name"].isin(opp_filter)]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            filtered = filtered[(filtered["match_date"] >= start_d) & (filtered["match_date"] <= end_d)]

        if search_query.strip():
            q = search_query.strip().lower()
            tags_str = filtered["tags"].apply(lambda items: " ".join(items).lower())
            mask = (
                filtered["opponent_name"].str.lower().str.contains(q, regex=False)
                | filtered["scoreline"].str.lower().str.contains(q, regex=False)
                | filtered["tactical_note"].str.lower().str.contains(q, regex=False)
                | tags_str.str.contains(q, regex=False)
            )
            filtered = filtered[mask]

        st.caption(f"Showing **{len(filtered)}** of **{len(df)}** logged matches")

        display_df = filtered[
            [
                "match_date",
                "opponent_name",
                "result",
                "scoreline",
                "court_surface",
                "match_type",
                "serve_rating",
                "mental_focus",
                "physical_energy",
                "first_serve_pct",
                "double_faults",
                "tags",
                "tactical_note",
            ]
        ].copy()
        display_df["tags"] = display_df["tags"].apply(lambda items: ", ".join(items) if items else "")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "match_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "opponent_name": st.column_config.TextColumn("Opponent"),
                "result": st.column_config.TextColumn("Result"),
                "scoreline": st.column_config.TextColumn("Scoreline"),
                "court_surface": st.column_config.TextColumn("Surface"),
                "match_type": st.column_config.TextColumn("Type"),
                "serve_rating": st.column_config.ProgressColumn("Serve", min_value=1, max_value=10, format="%d"),
                "mental_focus": st.column_config.ProgressColumn("Mental", min_value=1, max_value=10, format="%d"),
                "physical_energy": st.column_config.ProgressColumn("Energy", min_value=1, max_value=10, format="%d"),
                "first_serve_pct": st.column_config.NumberColumn("1st Srv %", format="%.1f%%"),
                "double_faults": st.column_config.NumberColumn("DF", format="%d"),
                "tags": st.column_config.TextColumn("Weapon / Leak"),
                "tactical_note": st.column_config.TextColumn("Tactical Note"),
            },
        )

        with st.expander("✏️ Edit or 🗑️ Delete a Logged Match", expanded=False):
            option_map: dict[str, pd.Series] = {}
            for _, row in filtered.iterrows():
                short_id = str(row["id"])[:8]
                score_part = f" ({row['scoreline']})" if row["scoreline"] else ""
                label = (
                    f"{row['match_date']} | vs {row['opponent_name']} | "
                    f"{row['result']}{score_part} | {row['court_surface']} [{short_id}]"
                )
                option_map[label] = row

            if not option_map:
                st.warning("No matches match your current filters above.")
            else:
                chosen_label = st.selectbox("Select Match to Edit or Delete", options=list(option_map.keys()))
                target = option_map[chosen_label]
                target_id = str(target["id"])

                with st.form(f"edit_match_form_{target_id}"):
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        edit_date = st.date_input("Match Date", value=target["match_date"])
                    with ec2:
                        edit_opp = st.text_input("Opponent Name", value=str(target["opponent_name"]))
                    with ec3:
                        edit_score = st.text_input("Scoreline", value=str(target["scoreline"]))

                    ec4, ec5, ec6 = st.columns(3)
                    with ec4:
                        res_idx = RESULTS.index(target["result"]) if target["result"] in RESULTS else 0
                        edit_res = st.selectbox("Result", options=RESULTS, index=res_idx)
                    with ec5:
                        surf_idx = (
                            COURT_SURFACES.index(target["court_surface"])
                            if target["court_surface"] in COURT_SURFACES
                            else 0
                        )
                        edit_surf = st.selectbox("Surface", options=COURT_SURFACES, index=surf_idx)
                    with ec6:
                        type_idx = (
                            MATCH_TYPES.index(target["match_type"])
                            if target["match_type"] in MATCH_TYPES
                            else 0
                        )
                        edit_type = st.selectbox("Match Type", options=MATCH_TYPES, index=type_idx)

                    es1, es2, es3 = st.columns(3)
                    with es1:
                        edit_serve = st.slider("Serve Rating", 1, 10, int(target["serve_rating"]))
                    with es2:
                        edit_mental = st.slider("Mental Focus", 1, 10, int(target["mental_focus"]))
                    with es3:
                        edit_energy = st.slider("Physical Energy", 1, 10, int(target["physical_energy"]))

                    ed1, ed2 = st.columns(2)
                    with ed1:
                        cur_1st = None if pd.isna(target["first_serve_pct"]) else float(target["first_serve_pct"])
                        edit_1st = st.number_input(
                            "1st Serve %",
                            min_value=0.0,
                            max_value=100.0,
                            value=cur_1st,
                            step=1.0,
                            format="%.1f",
                        )
                    with ed2:
                        cur_df = None if pd.isna(target["double_faults"]) else int(target["double_faults"])
                        edit_df = st.number_input(
                            "Double Faults",
                            min_value=0,
                            max_value=50,
                            value=cur_df,
                            step=1,
                        )

                    existing_tags = [t for t in target["tags"] if t in TACTICAL_TAGS]
                    edit_tags = st.multiselect("Weapon / Leak Tags", options=TACTICAL_TAGS, default=existing_tags)
                    edit_note = st.text_area("Tactical Note", value=str(target["tactical_note"]), height=80)

                    save_edit = st.form_submit_button("💾 Save Changes", type="primary")
                    if save_edit:
                        if not edit_opp.strip():
                            st.error("Opponent name cannot be empty.")
                        else:
                            updates = {
                                "match_date": edit_date.isoformat(),
                                "opponent_name": edit_opp.strip(),
                                "result": edit_res,
                                "scoreline": edit_score.strip() if edit_score.strip() else None,
                                "court_surface": edit_surf,
                                "match_type": edit_type,
                                "serve_rating": int(edit_serve),
                                "mental_focus": int(edit_mental),
                                "physical_energy": int(edit_energy),
                                "first_serve_pct": float(edit_1st) if edit_1st is not None else None,
                                "double_faults": int(edit_df) if edit_df is not None else None,
                                "tags": list(edit_tags),
                                "tactical_note": edit_note.strip() if edit_note.strip() else None,
                            }
                            try:
                                update_match(target_id, updates)
                                st.toast("Match updated!", icon="✅")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Error updating match: {exc}")

                st.markdown("---")
                del_col1, del_col2 = st.columns([2, 1])
                with del_col1:
                    confirm_del = st.checkbox(
                        f"Confirm permanent deletion of match vs {target['opponent_name']} ({target['match_date']})",
                        key=f"confirm_del_{target_id}",
                    )
                with del_col2:
                    if st.button("🗑️ Delete Match", type="secondary", use_container_width=True):
                        if not confirm_del:
                            st.warning("Check the confirmation box first.")
                        else:
                            try:
                                delete_match(target_id)
                                st.toast("Match deleted.", icon="🗑️")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Error deleting match: {exc}")

    st.divider()

    st.subheader("📥 Google Sheets CSV Migration Tool")
    st.caption(
        "Export your existing Google Sheets tennis log via **File → Download → Comma Separated Values (.csv)** "
        "and upload it below to bulk-import your match history into Supabase."
    )

    sample_csv = (
        "Date,Opponent,Match Type,Surface,Result,Scoreline,Serve,Mental,Energy,1st Serve %,Double Faults,Tags,Tactical Note\n"
        "2026-10-12,Marcus Vance,League,Hard,Win,\"6-4, 3-6, 10-7\",8,8,7,65.0,2,\"Forehand Winner, Return Depth\",Attacked second serves early\n"
        "2026-10-19,Elena Rostova,Tournament,Clay,Loss,\"4-6, 5-7\",6,7,6,58.5,5,\"Backhand Errors\",Need higher margin cross-court\n"
    )
    st.download_button(
        label="📄 Download Sample Google Sheets CSV Template",
        data=sample_csv,
        file_name="tennis_match_log_template.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader("Upload Google Sheets CSV Export", type=["csv"])
    if uploaded_csv is not None:
        try:
            raw_csv_df = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Could not read CSV file: {exc}")
            return

        if raw_csv_df.empty:
            st.warning("The uploaded CSV file contains no rows.")
            return

        st.markdown(f"**Detected {len(raw_csv_df)} rows and {len(raw_csv_df.columns)} columns.** Map your columns below:")

        csv_cols = list(raw_csv_df.columns)
        none_option = "-- Not in CSV (Use Default) --"
        mapping_options = [none_option] + csv_cols

        def auto_detect(keywords: list[str]) -> int:
            for idx, col in enumerate(csv_cols):
                col_low = str(col).strip().lower()
                if any(kw in col_low for kw in keywords):
                    return idx + 1
            return 0

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            col_date = st.selectbox("Match Date *", mapping_options, index=auto_detect(["date", "timestamp", "day"]))
            col_opp = st.selectbox("Opponent Name *", mapping_options, index=auto_detect(["opponent", "player", "name"]))
            col_res = st.selectbox("Result (Win/Loss) *", mapping_options, index=auto_detect(["result", "outcome", "w/l"]))
            col_score = st.selectbox("Scoreline", mapping_options, index=auto_detect(["score"]))
        with mc2:
            col_surf = st.selectbox("Court Surface", mapping_options, index=auto_detect(["surface", "court"]))
            col_type = st.selectbox("Match Type", mapping_options, index=auto_detect(["type", "format", "event"]))
            col_serve = st.selectbox("Serve Rating (1–10)", mapping_options, index=auto_detect(["serve"]))
            col_mental = st.selectbox("Mental Focus (1–10)", mapping_options, index=auto_detect(["mental", "focus"]))
        with mc3:
            col_energy = st.selectbox("Physical Energy (1–10)", mapping_options, index=auto_detect(["energy", "physical", "fitness"]))
            col_1st = st.selectbox("1st Serve %", mapping_options, index=auto_detect(["1st", "first"]))
            col_df = st.selectbox("Double Faults", mapping_options, index=auto_detect(["double", "df", "fault"]))
            col_tags = st.selectbox("Weapon / Leak Tags", mapping_options, index=auto_detect(["tag", "weapon", "leak"]))

        col_note = st.selectbox("Tactical Note", mapping_options, index=auto_detect(["note", "comment", "takeaway"]))

        if col_date == none_option or col_opp == none_option or col_res == none_option:
            st.warning("Map at least **Match Date**, **Opponent Name**, and **Result** to preview and import.")
            return

        normalized_records: list[dict[str, Any]] = []
        skipped_rows = 0

        parsed_dates = pd.to_datetime(raw_csv_df[col_date], errors="coerce")

        for idx, row in raw_csv_df.iterrows():
            dt_val = parsed_dates.loc[idx]
            opp_val = str(row[col_opp]).strip() if pd.notna(row[col_opp]) else ""
            if pd.isna(dt_val) or not opp_val or opp_val.lower() == "nan":
                skipped_rows += 1
                continue

            res_val = normalize_result_value(row[col_res]) if pd.notna(row[col_res]) else "Loss"
            score_val = (
                str(row[col_score]).strip()
                if col_score != none_option and pd.notna(row[col_score]) and str(row[col_score]).strip()
                else None
            )
            surf_val = (
                normalize_surface_value(row[col_surf])
                if col_surf != none_option and pd.notna(row[col_surf])
                else "Hard"
            )
            type_val = (
                normalize_match_type_value(row[col_type])
                if col_type != none_option and pd.notna(row[col_type])
                else "Practice Match"
            )

            def clean_rating(mapped_col: str) -> int:
                if mapped_col == none_option or pd.isna(row[mapped_col]):
                    return 7
                num = pd.to_numeric(row[mapped_col], errors="coerce")
                if pd.isna(num):
                    return 7
                return int(min(10, max(1, round(float(num)))))

            serve_val = clean_rating(col_serve)
            mental_val = clean_rating(col_mental)
            energy_val = clean_rating(col_energy)

            first_srv_val = None
            if col_1st != none_option and pd.notna(row[col_1st]):
                raw_1st = str(row[col_1st]).replace("%", "").strip()
                num_1st = pd.to_numeric(raw_1st, errors="coerce")
                if not pd.isna(num_1st):
                    clamped_1st = min(100.0, max(0.0, float(num_1st)))
                    first_srv_val = round(clamped_1st, 2)

            df_val = None
            if col_df != none_option and pd.notna(row[col_df]):
                num_df = pd.to_numeric(row[col_df], errors="coerce")
                if not pd.isna(num_df):
                    df_val = int(max(0, round(float(num_df))))

            tags_list: list[str] = []
            if col_tags != none_option and pd.notna(row[col_tags]):
                raw_tags = str(row[col_tags]).strip()
                if raw_tags and raw_tags.lower() != "nan":
                    tags_list = [t.strip() for t in re.split(r"[,;|]", raw_tags) if t.strip()]

            note_val = (
                str(row[col_note]).strip()
                if col_note != none_option and pd.notna(row[col_note]) and str(row[col_note]).strip()
                else None
            )

            normalized_records.append(
                {
                    "match_date": dt_val.strftime("%Y-%m-%d"),
                    "opponent_name": opp_val,
                    "match_type": type_val,
                    "court_surface": surf_val,
                    "result": res_val,
                    "scoreline": score_val,
                    "serve_rating": serve_val,
                    "mental_focus": mental_val,
                    "physical_energy": energy_val,
                    "first_serve_pct": first_srv_val,
                    "double_faults": df_val,
                    "tags": tags_list,
                    "tactical_note": note_val,
                }
            )

        if skipped_rows > 0:
            st.warning(f"{skipped_rows} row(s) will be skipped due to missing Date or Opponent Name.")

        if normalized_records:
            st.markdown("**Preview of Normalized Records (First 5 Rows):**")
            st.dataframe(pd.DataFrame(normalized_records[:5]), use_container_width=True, hide_index=True)

            if st.button(
                f"🚀 Import {len(normalized_records)} Matches into Supabase",
                type="primary",
                use_container_width=True,
            ):
                try:
                    count = bulk_insert_matches(normalized_records)
                    st.toast(f"Imported {count} matches!", icon="🎉")
                    st.success(f"Successfully imported **{count}** historical matches into Supabase.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Bulk import failed: {exc}")


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    require_pin_auth()

    head_left, head_right = st.columns([5, 1])
    with head_left:
        st.title("🎾 CourtIQ — Personal Tennis Analytics")
    with head_right:
        st.write("")
        if st.button("🔒 Lock", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    try:
        matches_df = load_matches_df()
    except Exception as exc:
        st.error(
            f"Unable to connect to Supabase (`personal_matches`). Verify `SUPABASE_URL`, "
            f"`SUPABASE_KEY`, and that the SQL table setup script has been executed.\n\nDetails: {exc}"
        )
        st.stop()

    tab_log, tab_analytics, tab_scouting, tab_manage = st.tabs(
        [
            "🎾 Log New Match",
            "📊 Performance Analytics",
            "🆚 Opponent Scouting",
            "📋 Match Log & CSV Importer",
        ]
    )

    with tab_log:
        render_log_match_tab(matches_df)

    with tab_analytics:
        render_analytics_tab(matches_df)

    with tab_scouting:
        render_scouting_tab(matches_df)

    with tab_manage:
        render_log_and_importer_tab(matches_df)


if __name__ == "__main__":
    main()
