import streamlit as st
import pandas as pd
from datetime import date
from supabase import create_client, Client

# --- PAGE CONFIG ---
st.set_page_config(page_title="CourtIQ | Tennis Analytics", page_icon="🎾", layout="wide")

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
        st.title("🎾 CourtIQ Login")
        pin_input = st.text_input("Enter your access PIN", type="password")
        if st.button("Unlock"):
            if pin_input == st.secrets["APP_PIN"]:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect PIN. Try again.")
        st.stop()

check_pin()

# --- HELPER FUNCTIONS ---
def load_data():
    response = supabase.table("personal_matches").select("*").order("match_date", desc=True).execute()
    return pd.DataFrame(response.data)

def insert_match(record: dict):
    supabase.table("personal_matches").insert(record).execute()

# --- MAIN APP ---
st.title("🎾 CourtIQ — Personal Tennis Tracker")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "➕ Log Session", "📁 Match Log & AI Export"])

df = load_data()

# --- TAB 1: DASHBOARD ---
with tab1:
    if df.empty:
        st.info("No sessions logged yet. Head over to 'Log Session' or import a CSV!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_sessions = len(df)
        matches_only = df[df["match_type"].isin(["Singles Match", "Doubles Match", "Practice Match"])]
        wins = len(matches_only[matches_only["result"] == "Win"])
        win_rate = f"{(wins / len(matches_only) * 100):.0f}%" if len(matches_only) > 0 else "N/A"
        wall_sessions = len(df[df["match_type"] == "Wall Session"])

        col1.metric("Total Sessions", total_sessions)
        col2.metric("Match Win Rate", win_rate)
        col3.metric("Wall Sessions", wall_sessions)
        col4.metric("Avg Focus (1-5)", f"{df['mental_focus'].mean():.1f}" if "mental_focus" in df else "N/A")

        st.subheader("Recent Sessions")
        st.dataframe(df[["match_date", "match_type", "opponent", "result", "score", "tactical_note"]], use_container_width=True)

# --- TAB 2: LOG SESSION (FAST 30-SEC CHECK-IN) ---
with tab2:
    st.subheader("Log a Match, Drill, or Wall Session")
    with st.form("log_session_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        session_date = c1.date_input("Date", value=date.today())
        match_type = c2.selectbox(
            "Session Type",
            ["Singles Match", "Doubles Match", "Practice Match", "Wall Session", "Drills / Coaching"]
        )
        surface = c3.selectbox("Surface", ["Hard", "Clay", "Grass", "Wall / Concrete"])

        c4, c5, c6 = st.columns(3)
        opponent = c4.text_input("Opponent / Focus (e.g., Wall, John)", value="Wall" if match_type == "Wall Session" else "")
        result = c5.selectbox("Result", ["N/A (Training)", "Win", "Loss"])
        score = c6.text_input("Score or Duration (e.g., 6-4 6-3 or 45 mins)")

        st.markdown("---")
        st.markdown("**Quick Feel & Ratings (1 = Low, 5 = Peak)**")
        r1, r2, r3 = st.columns(3)
        serve_rating = r1.slider("Stroke / Serve Feel", 1, 5, 3)
        mental_focus = r2.slider("Mental Focus", 1, 5, 3)
        energy_level = r3.slider("Energy / Footwork", 1, 5, 3)

        st.markdown("---")
        st.markdown("**Fast AI Reflection (Keep it brief)**")
        t1, t2 = st.columns(2)
        main_weapon = t1.selectbox(
            "What worked best today?",
            ["Forehand", "Backhand", "Serve", "Return", "Footwork / Consistency", "Net Play", "Mental Composure"]
        )
        main_leak = t2.selectbox(
            "Primary adjustment for next time",
            ["Forehand consistency", "Backhand depth/control", "Serve percentage", "Footwork / Split step", "Patience / Shot selection", "None"]
        )
        tactical_note = st.text_area(
            "How did you feel & key adjustment? (Dictate or write 1–2 sentences for your AI review)",
            placeholder="e.g., Forehand felt solid crosscourt against the wall, need to bend knees more on low backhands next time."
        )

        submitted = st.form_submit_button("Save Session", use_container_width=True)
        if submitted:
            formatted_note = f"[Best: {main_weapon} | Adjust: {main_leak}] {tactical_note}".strip()
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
                "tactical_note": formatted_note
            }
            insert_match(new_record)
            st.success("Session saved to Supabase!")
            st.rerun()

# --- TAB 3: MATCH LOG, CSV IMPORTER & AI EXPORT ---
with tab3:
    st.subheader("Export Data for AI Review (NotebookLM / Gemini)")
    if not df.empty:
        csv_export = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download All Sessions as CSV for AI",
            data=csv_export,
            file_name=f"courtiq_tennis_log_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("Log or import sessions first to enable CSV export.")

    st.markdown("---")
    st.subheader("Full Session History")
    st.dataframe(df, use_container_width=True)
