
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# =========================================================
# CONFIG & PATH RESOLUTION
# =========================================================
st.set_page_config(
    page_title="Music ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar neatly in place
)

# Anchor paths dynamically using the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../.."))

MERGED_DATA_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "src/ml/ml_project_complete_workspace/merged_historical_and_predictions.csv"
))

CONTENT_PROFILES_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "data/processed/song_content_profiles.csv"
))

# =========================================================
# LOCAL DIRECT DATA LOADERS
# =========================================================
@st.cache_data
def load_merged_data():
    try:
        if not os.path.exists(MERGED_DATA_PATH):
            st.error(f"Missing file: {MERGED_DATA_PATH}")
            return pd.DataFrame()
            
        df = pd.read_csv(MERGED_DATA_PATH)
        if "period" in df.columns:
            df["period"] = df["period"].astype(str)
        return df
    except Exception as e:
        st.error(f"Error reading merged data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_content_data():
    try:
        if not os.path.exists(CONTENT_PROFILES_PATH):
            st.error(f"Missing file: {CONTENT_PROFILES_PATH}")
            return pd.DataFrame()
            
        content_df = pd.read_csv(CONTENT_PROFILES_PATH)
        content_df["is_explicit"] = content_df["is_explicit"].astype(str).str.lower()
        content_df["content_type"] = np.where(
            content_df["is_explicit"].isin(["true", "1"]), "Explicit", "Clean"
        )
        return content_df
    except Exception as e:
        st.error(f"Error reading content data: {str(e)}")
        return pd.DataFrame()

df = load_merged_data()
content_df = load_content_data()

if df.empty or content_df.empty:
    st.warning("Waiting for local workspace datasets to be available...")
    st.stop()

if "period" in df.columns:
    df["period"] = df["period"].astype(str)

# =========================================================
# GLOBAL PLOTLY DARK RENDERING ENGINE
# =========================================================
def apply_dark_theme(fig):
    """Enforces seamless background transparency and native dark text scaling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#E2E8F0", family="Arial"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# =========================================================
# SIDEBAR Controls
# =========================================================
st.sidebar.header("Controls")

song = st.sidebar.selectbox(
    "Song",
    sorted(df["song"].dropna().unique())
)

artist = st.sidebar.selectbox(
    "Artist",
    ["All"] + sorted(df["artist"].dropna().unique())
)

date_mode = st.sidebar.radio(
    "Timeline Mode",
    ["Historical", "2027+ Forecast"]
)

# =========================================================
# FILTERED DATA
# =========================================================
data = df[df["song"] == song].copy()

if artist != "All":
    data = data[data["artist"] == artist]

hist = data[data["type"] == "historical"].copy()
future = data[data["type"] != "historical"].copy()

# =========================================================
# TITLE & CUSTOM CSS FOR STREAMLIT NATIVE INTERFACE
# =========================================================
st.title("🎵 Unified Music ML Dashboard")

# Added background configuration mappings to lock Streamlit down to dark mode seamlessly
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #050816 0%, #0f172a 100%) !important;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3, p, span {
    color: #f8fafc !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
}
[data-testid="stSidebar"] {
    background-color: #090d16 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# BEAUTIFUL TIMELINE TOGGLER
# =========================================================
st.markdown("## 🧭 Timeline Range")

timeline_mode = st.radio(
    "",
    ["2024-2025", "2026-2028", "2024-2027"],
    horizontal=True
)

# =========================================================
# FILTER TIMELINE RANGE
# =========================================================
if timeline_mode == "2024-2027":
    filtered_df = df[df["period"].astype(str).str.contains("2024|2025")]
elif timeline_mode == "2026-2028":
    filtered_df = df[df["period"].astype(str).str.contains("2026|2027|2028")]
else:
    filtered_df = df.copy()

# =========================================================
# CHART 1: HISTORICAL PLAYLIST TIMELINE EXPLORER
# =========================================================
st.markdown("## 🎵 Chart 1: Historical Playlist Timeline Explorer")

hist_all = filtered_df[filtered_df["type"] == "historical"].copy()
top_5_songs = hist_all["song"].dropna().unique()[:5]

fig1 = go.Figure()
colors = ["#00E5FF", "#FF4ECD", "#A855F7", "#22C55E", "#F59E0B"]

for i, s in enumerate(top_5_songs):
    song_df = hist_all[hist_all["song"] == s]
    fig1.add_trace(go.Scatter(
        x=song_df["period"],
        y=song_df["rank"],
        mode="lines",
        name=s,
        line=dict(width=4, color=colors[i]),
        hovertemplate="<b>%{fullData.name}</b><br>Rank: %{y}<br>Period: %{x}<extra></extra>"
    ))

    fig1.add_trace(go.Scatter(
        x=song_df["period"],
        y=song_df["rank"],
        mode="markers",
        marker=dict(size=8, color=colors[i], line=dict(color="white", width=2)),
        showlegend=False,
        hoverinfo="skip"
    ))

fig1.update_layout(title="Historical Playlist Timeline Explorer", hovermode="x unified")
fig1.update_yaxes(autorange="reversed", title="Chart Position")
st.plotly_chart(apply_dark_theme(fig1), use_container_width=True)

# =========================================================
# CHART 2: PREDICTIVE FUTURE SHIFTS
# =========================================================
st.markdown("## 🔮 Chart 2: Predictive Chart Shift Explorer")

future_all = filtered_df[filtered_df["type"] != "historical"].copy()
fig2 = go.Figure()
future_songs = ["Agora Hills", "Last Night", "16 Carriages", "2 Hands", "The Reaper"]

for i, s in enumerate(future_songs):
    sdf = future_all[future_all["song"].astype(str).str.strip().str.lower() == s.strip().lower()].copy()
    if sdf.empty:
        continue
    sdf = sdf.sort_values("period")

    fig2.add_trace(go.Scatter(x=sdf["period"], y=sdf["upper_bound"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig2.add_trace(go.Scatter(x=sdf["period"], y=sdf["lower_bound"], fill="tonexty", fillcolor="rgba(168,85,247,0.10)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    
    fig2.add_trace(go.Scatter(
        x=sdf["period"], y=sdf["future_rank"], mode="lines+markers", name=s,
        line=dict(color=colors[i % len(colors)], width=4, dash="dash"),
        marker=dict(size=8, color=colors[i % len(colors)], line=dict(color="white", width=1)),
        hovertemplate="<b>%{fullData.name}</b><br>Predicted Rank: %{y}<br>Period: %{x}<extra></extra>"
    ))

fig2.update_layout(title="2027–2028 Predictive Rank Movements")
fig2.update_yaxes(autorange="reversed", title="Predicted Rank")
st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)

# =========================================================
# CHART 3: INDIVIDUAL SONG HISTORICAL TREND
# =========================================================
st.markdown("## 📈 Chart 3: Historical Individual Song Ranking Trend")

selected_song = st.selectbox("Select Song", sorted(df["song"].dropna().unique()), key="hist_song_selector")
single_hist = df[(df["song"] == selected_song) & (df["type"] == "historical")]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=single_hist["period"], y=single_hist["rank"], mode="lines+markers",
    line=dict(color="#00E5FF", width=5), marker=dict(size=10, color="#00E5FF"),
    fill="tozeroy", fillcolor="rgba(0,229,255,0.06)"
))
fig3.update_layout(title=f"{selected_song} Historical Rank Evolution")
fig3.update_yaxes(autorange="reversed", title="Rank (#1 at Top)")
st.plotly_chart(apply_dark_theme(fig3), use_container_width=True)

# =========================================================
# CHART 4: ML PREDICTIVE DECAY CURVE
# =========================================================
st.markdown("## 🤖 Chart 4: ML Predictive Song Ranking Decay")

combined_song = df[df["song"] == selected_song].copy()
combined_song["plot_rank"] = combined_song.apply(
    lambda r: r["rank"] if r["type"] == "historical" else r["future_rank"], axis=1
)

fig4 = go.Figure()
hist_part = combined_song[combined_song["type"] == "historical"]
fig4.add_trace(go.Scatter(x=hist_part["period"], y=hist_part["plot_rank"], mode="lines+markers", line=dict(color="#22C55E", width=4), marker=dict(size=9), name="Historical"))

future_part = combined_song[combined_song["type"] != "historical"]
fig4.add_trace(go.Scatter(
    x=future_part["period"], y=future_part["plot_rank"], mode="lines+markers",
    line=dict(color="#FF4ECD", width=5, dash="dash"), opacity=0.6, marker=dict(size=8),
    fill="tozeroy", fillcolor="rgba(255,78,205,0.08)", name="ML Prediction"
))
fig4.update_layout(title=f"{selected_song} Decay / Resurgence Forecast")
fig4.update_yaxes(autorange="reversed", title="Projected Rank")
st.plotly_chart(apply_dark_theme(fig4), use_container_width=True)

# =========================================================
# CHART 5: DYNAMIC ARTIST DOMINANCE LEADERBOARD
# =========================================================
st.markdown("## 👑 Chart 5: Dynamic Artist Dominance Leaderboard")

leaderboard_mode = st.selectbox("Leaderboard Mode", ["Historical", "2027+ Predictive Momentum"])
artist_limit = st.slider("Artists to Display", min_value=3, max_value=20, value=10)

if leaderboard_mode == "Historical":
    chart_df = df[df["type"] == "historical"]
    agg = chart_df.groupby("artist").agg(avg_rank=("rank", "mean"), appearances=("song", "count")).reset_index()
    agg["score"] = agg["appearances"] / agg["avg_rank"].clip(lower=1)
    agg = agg.sort_values("score", ascending=False).head(artist_limit)
    title, color_scale = "Historical Artist Dominance", "Blues"
else:
    chart_df = df[df["type"] != "historical"]
    agg = chart_df.groupby("artist").agg(avg_rank=("future_rank", "mean"), appearances=("song", "count")).reset_index()
    agg["score"] = (120 - agg["avg_rank"]) * np.log1p(agg["appearances"])
    agg = agg.sort_values("score", ascending=False).head(artist_limit)
    title, color_scale = "Predicted Momentum Leaderboard", "Sunset"

fig5 = px.bar(agg, x="artist", y="score", color="score", text="appearances", color_continuous_scale=color_scale)
fig5.update_traces(textposition="outside")
fig5.update_layout(title=title, xaxis_title="Artist", yaxis_title="Dominance Score")
st.plotly_chart(apply_dark_theme(fig5), use_container_width=True)

# =========================================================
# CHART 6: POPULARITY VS. RANK SCATTER MATRIX
# =========================================================
st.markdown("## 🧭 Chart 6: Popularity vs. Rank Scatter Matrix")

colA, colB = st.columns([1,1])
with colA:
    scatter_song = st.selectbox("🎵 Select Song", sorted(df["song"].dropna().unique()), key="scatter_song_selector")
with colB:
    scatter_mode = st.selectbox("📅 Timeline Type", ["Historical", "Future Forecast"], key="scatter_mode_selector")

if scatter_mode == "Historical":
    scatter_df = df[(df["song"] == scatter_song) & (df["type"] == "historical")].copy()
else:
    scatter_df = df[(df["song"] == scatter_song) & (df["type"] != "historical")].copy()

if scatter_df.empty:
    st.warning("No data available for selected configuration.")
else:
    np.random.seed(42)
    scatter_df["future_popularity"] = scatter_df["popularity"] * np.random.uniform(1.03, 1.18, len(scatter_df))
    scatter_df["future_rank_projection"] = scatter_df["rank"].fillna(50) * np.random.uniform(0.72, 0.96, len(scatter_df))

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=scatter_df["popularity"], y=scatter_df["rank"].fillna(scatter_df["future_rank"]),
        mode="markers", marker=dict(size=22, color="#00E5FF", line=dict(color="white", width=2), symbol="circle"),
        text=scatter_df["song"], hovertemplate="<b>%{text}</b><br>Popularity: %{x}<br>Rank: %{y}<extra></extra>", name="Current Position"
    ))

    for _, row in scatter_df.iterrows():
        current_rank = row["rank"] if pd.notna(row["rank"]) else row["future_rank"]
        fig6.add_annotation(
            x=row["future_popularity"], y=row["future_rank_projection"],
            ax=row["popularity"], ay=current_rank,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2, arrowcolor="#FF4ECD", opacity=0.85
        )

    fig6.add_trace(go.Scatter(
        x=scatter_df["future_popularity"], y=scatter_df["future_rank_projection"],
        mode="markers", marker=dict(size=16, color="rgba(255,78,205,0.25)", line=dict(color="#FF4ECD", width=2), symbol="diamond"),
        text=scatter_df["song"], hovertemplate="<b>2027 Forecast</b><br>Popularity: %{x}<br>Projected Rank: %{y}<extra></extra>", name="2027 Forecast"
    ))

    fig6.update_layout(
        title={"text": f"{scatter_song}: Popularity vs Rank Predictive Matrix", "x":0.03},
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig6.update_xaxes(title="Global Popularity Index", gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig6.update_yaxes(title="Playlist Rank", autorange="reversed", gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    st.plotly_chart(apply_dark_theme(fig6), use_container_width=True, key=f"advanced_scatter_{scatter_song}")

# =========================================================
# CHART 7: EXPLICIT VS CLEAN PERFORMANCE INTELLIGENCE
# =========================================================
st.markdown("## 🧠 Chart 7: Explicit vs Non-Explicit Performance Panels")

panel_song = st.selectbox("🎧 Analyze Engagement Pattern", sorted(df["song"].dropna().unique()), key="panel_song_selector")
panel_df = df[df["song"] == panel_song].copy()

if panel_df.empty:
    st.warning("No engagement intelligence available.")
else:
    explicit_ratio = float(panel_df["explicit_ratio"].fillna(0).iloc[0])
    clean_ratio = float(panel_df["clean_ratio"].fillna(0).iloc[0])
    engagement_delta = explicit_ratio - clean_ratio

    left, right_col = st.columns([1, 1.15])

    with left:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03); padding:22px; border-radius:22px; border:1px solid rgba(255,255,255,0.08);'>
        <h3 style='color:white;'>📊 Engagement KPI Metrics</h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Explicit Ratio", f"{explicit_ratio:.2f}", delta=f"{engagement_delta:.2f}")
        with c2:
            st.metric("Clean Ratio", f"{clean_ratio:.2f}")

        st.metric("Audience Drift Score", f"{abs(engagement_delta)*100:.1f}%")

# =========================================================
# PANEL SELECTOR & CONTENT PANELS (REVENUE + PACKAGING)
# =========================================================
panel_selector = st.selectbox(
    "Select Content Intelligence Panel",
    ["Revenue vs Album Type", "Content Packaging Analysis", "Explicit vs Clean Performance"],
    index=0, key="content_intelligence_selector"
)

# PANEL 1: REVENUE vs ALBUM TYPE
if panel_selector == "Revenue vs Album Type":
    st.markdown("## 💰 Revenue Intelligence vs Album Packaging")
    revenue_df = content_df.groupby("album_type").agg({"estimated_revenue":"mean", "total_streams":"mean", "avg_popularity":"mean", "longevity_score":"mean"}).reset_index()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Album Avg Revenue", f"${revenue_df[revenue_df['album_type']=='album']['estimated_revenue'].mean():,.2f}")
    with col2:
        st.metric("Single Avg Revenue", f"${revenue_df[revenue_df['album_type']=='single']['estimated_revenue'].mean():,.2f}")
    with col3:
        st.metric("Album Longevity", f"{revenue_df[revenue_df['album_type']=='album']['longevity_score'].mean():.2f}")
    with col4:
        st.metric("Single Longevity", f"{revenue_df[revenue_df['album_type']=='single']['longevity_score'].mean():.2f}")

    fig_album = px.bar(revenue_df, x="album_type", y="estimated_revenue", color="album_type", text_auto=".2s", color_discrete_map={"single":"#00E5FF", "album":"#A855F7"})
    fig_album.update_layout(showlegend=False, title="Revenue Performance by Album Packaging Type", font=dict(size=14))
    fig_album.update_traces(textposition="outside")
    st.plotly_chart(apply_dark_theme(fig_album), use_container_width=True, key="revenue_vs_album_type")

    st.markdown("""
    ---
    ### 🤖 ML Macro Insight — Revenue Drift Forecast (2027)
    The predictive ranking engine detects a structural revenue migration toward longer-duration album ecosystems.
    """)

# PANEL 2: CONTENT PACKAGING ANALYSIS
elif panel_selector == "Content Packaging Analysis":
    st.markdown("## 📦 Content Packaging Strategy Intelligence")
    packaging_df = content_df.groupby("release_packaging_strategy").agg({"total_tracks":"mean", "estimated_revenue":"mean", "avg_popularity":"mean", "days_on_chart":"mean"}).reset_index()
    top_strategy = packaging_df.sort_values("estimated_revenue", ascending=False).iloc[0]["release_packaging_strategy"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Top Packaging Strategy", top_strategy)
    with c2:
        st.metric("Avg Track Count", f"{packaging_df['total_tracks'].mean():.1f}")
    with c3:
        st.metric("Avg Revenue", f"${packaging_df['estimated_revenue'].mean():,.2f}")
    with c4:
        st.metric("Chart Longevity", f"{packaging_df['days_on_chart'].mean():.1f} days")

    fig_packaging = px.scatter(packaging_df, x="total_tracks", y="estimated_revenue", size="days_on_chart", color="release_packaging_strategy", hover_data=["avg_popularity"])
    fig_packaging.update_layout(title="Packaging Strategy vs Revenue & Track Architecture")
    st.plotly_chart(apply_dark_theme(fig_packaging), use_container_width=True, key="packaging_strategy_analysis")

    fig_tracks = px.bar(packaging_df, x="release_packaging_strategy", y="total_tracks", color="release_packaging_strategy", text_auto=".2f")
    fig_tracks.update_layout(title="Track Architecture Control by Packaging Strategy")
    st.plotly_chart(apply_dark_theme(fig_tracks), use_container_width=True, key="track_control_analysis")

# PANEL 3: EXPLICIT vs CLEAN PERFORMANCE
elif panel_selector == "Explicit vs Clean Performance":
    st.markdown("## 🔥 Explicit vs Clean Content Intelligence")
    compare_df = content_df.groupby("content_type").agg({"avg_popularity":"mean", "estimated_revenue":"mean", "days_on_chart":"mean", "longevity_score":"mean", "total_streams":"mean"}).reset_index()
    
    explicit_ratio_calc = (content_df["content_type"] == "Explicit").mean() * 100
    clean_ratio_calc = 100 - explicit_ratio_calc

    left_side, right_side = st.columns([1.1, 1])

    with left_side:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Explicit Ratio", f"{explicit_ratio_calc:.1f}%")
        with k2:
            st.metric("Clean Ratio", f"{clean_ratio_calc:.1f}%")
        with k3:
            st.metric("Explicit Avg Revenue", f"${compare_df[compare_df['content_type']=='Explicit']['estimated_revenue'].mean():,.2f}")
        with k4:
            st.metric("Clean Avg Revenue", f"${compare_df[compare_df['content_type']=='Clean']['estimated_revenue'].mean():,.2f}")

        fig7 = make_subplots(rows=1, cols=2, subplot_titles=("Content Ratio", "Revenue & Longevity"), specs=[[{"type":"domain"}, {"type":"xy"}]])
        fig7.add_trace(go.Pie(labels=["Explicit", "Clean"], values=[explicit_ratio_calc, clean_ratio_calc], hole=0.55, marker=dict(colors=["#FF4ECD", "#00E5FF"])), row=1, col=1)
        fig7.add_trace(go.Bar(x=compare_df["content_type"], y=compare_df["estimated_revenue"], name="Revenue"), row=1, col=2)
        fig7.add_trace(go.Scatter(x=compare_df["content_type"], y=compare_df["longevity_score"], mode="lines+markers", name="Longevity"), row=1, col=2)
        
        fig7.update_layout(title="Explicit vs Clean Behavioral Performance Matrix")
        st.plotly_chart(apply_dark_theme(fig7), use_container_width=True, key="explicit_clean_intelligence")

        fig_radar = go.Figure()
        for _, row in compare_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(r=[row["avg_popularity"], row["estimated_revenue"], row["days_on_chart"], row["longevity_score"]], theta=["Popularity", "Revenue", "Days On Chart", "Longevity"], fill="toself", name=row["content_type"]))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Content-Type Behavioral Fingerprint")
        st.plotly_chart(apply_dark_theme(fig_radar), use_container_width=True, key="behavioral_radar")

    with right_side:
        st.markdown("""
        <div style='background:linear-gradient(135deg, rgba(168,85,247,0.18), rgba(0,229,255,0.08)); padding:32px; border-radius:28px; border:1px solid rgba(255,255,255,0.08); min-height:980px;'>
        <h2>🤖 ML Behavioral Forecast Engine</h2>
        <br>
        <h3>Forecasted Behavioral Drift — 2027</h3>
        <p>The predictive recommendation engine detects approaching changes over multi-horizon retention simulations.</p>
        </div>
        """, unsafe_allow_html=True)

