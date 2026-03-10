"""
Module 5: Student UI — Multi-Modal Online Learning Resource Finder
Tab-based Streamlit app covering all 5 modules from the problem statement.
"""

import streamlit as st
from PIL import Image
from nlp_ml_model import predict_topic
from image_dl_model import predict_image
from real_time_recommender import youtube_search
from recommendation_model import recommend, get_all_sources, get_all_topics, _load_resources
from visualization import (
    create_source_distribution_chart,
    create_level_distribution_chart,
    create_learning_path_chart,
    create_topic_heatmap,
)

st.set_page_config(page_title="MultiModal Learning Finder", layout="wide")

# ── Custom Styling ──
st.markdown("""
<style>
body { background-color: #0E1117; }
.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00C9A7, #845EC2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    text-align: center;
    color: #888;
    font-size: 17px;
    margin-bottom: 20px;
}
.resource-card {
    padding: 16px 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1E2228 0%, #262B33 100%);
    border: 1px solid #333;
    margin-bottom: 10px;
}
.resource-card h4 { margin: 0 0 6px 0; color: #E0E0E0; }
.resource-card p  { margin: 0; color: #999; font-size: 14px; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
    margin-right: 4px;
}
.badge-beginner     { background: #1B5E20; color: #A5D6A7; }
.badge-intermediate { background: #E65100; color: #FFCC80; }
.badge-advanced     { background: #B71C1C; color: #EF9A9A; }
.badge-youtube      { background: #CC0000; color: #FFCDD2; }
.badge-coursera     { background: #0056D2; color: #BBDEFB; }
.badge-udemy        { background: #A435F0; color: #E1BEE7; }
.badge-freecamp     { background: #0A0A23; color: #90CAF9; }
.badge-google       { background: #4285F4; color: #BBDEFB; }
.badge-edx          { background: #02262B; color: #80CBC4; }
.badge-khan         { background: #14BF96; color: #E0F2F1; }
.badge-default      { background: #37474F; color: #B0BEC5; }
.topic-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
    background: linear-gradient(90deg, #00C9A7, #845EC2);
    color: white;
    margin: 4px 2px;
}
.stat-box {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1E2228 0%, #262B33 100%);
    border: 1px solid #333;
}
.stat-box h2 { margin: 0; color: #00C9A7; }
.stat-box p  { margin: 4px 0 0; color: #999; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎓 Multi-Modal Online Learning Resource Finder</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-Powered Image + NLP Based Smart Learning Recommendation System</p>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Tabs matching the 5 modules
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔍 Find Resources",
    "📚 All Resources",
    "📊 Learning Dashboard",
])

# ── Source badge helper ──
SOURCE_BADGE_MAP = {
    "YouTube": "badge-youtube",
    "Coursera": "badge-coursera",
    "Udemy": "badge-udemy",
    "freeCodeCamp": "badge-freecamp",
    "Google": "badge-google",
    "edX": "badge-edx",
    "Khan Academy": "badge-khan",
}


def _source_badge(source: str) -> str:
    cls = SOURCE_BADGE_MAP.get(source, "badge-default")
    return f'<span class="badge {cls}">{source}</span>'


def _level_badge(level: str) -> str:
    cls = f"badge-{level.lower()}" if level else ""
    return f'<span class="badge {cls}">{level}</span>'


def _render_resource_card(row, topic_label: str = ""):
    """Render a single resource card."""
    level = row.get("level", "")
    source = row.get("source", "")
    url = row.get("url", "")
    desc = row.get("description", "")
    topic = row.get("topic", topic_label)

    link_html = f' <a href="{url}" target="_blank" style="color:#00C9A7;">🔗 Open</a>' if url else ""

    st.markdown(f"""
    <div class="resource-card">
        {_level_badge(level)} {_source_badge(source)}
        <h4>{desc}{link_html}</h4>
        <p>Topic: {topic}</p>
    </div>
    """, unsafe_allow_html=True)


def _render_youtube_card(video: dict, topic: str):
    """Render a YouTube video as a resource card."""
    yt_link = video["link"]
    vid_id = yt_link.split("v=")[-1] if "v=" in yt_link else ""
    thumb = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg" if vid_id else ""
    thumb_html = (
        f'<img src="{thumb}" style="width:120px;border-radius:8px;'
        f'margin-right:14px;vertical-align:middle;" />'
        if thumb else ""
    )

    st.markdown(f"""
    <div class="resource-card" style="display:flex;align-items:center;">
        {thumb_html}
        <div>
            <span class="badge badge-youtube">🎬 YouTube Live</span>
            <h4>{video['title']}
                <a href="{yt_link}" target="_blank" style="color:#00C9A7;">▶ Watch</a>
            </h4>
            <p>Real-time recommendation for: {topic}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Find Resources (Modules 1, 2, 3)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Image-Based Resource Detector")
        uploaded_image = st.file_uploader("Upload a study-related image", type=["jpg", "png", "jpeg"])

    with col2:
        st.subheader("📝 NLP Query Understanding")
        user_query = st.text_input(
            "Enter your learning query",
            placeholder="e.g. learn machine learning, build a website, deploy on AWS",
        )

    st.divider()

    # Filters
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        level_filter = st.selectbox("🎯 Difficulty Level", ["All", "Beginner", "Intermediate", "Advanced"])
    with fcol2:
        all_sources = get_all_sources()
        source_filter = st.multiselect("🌐 Filter by Platform", all_sources, default=[])

    if st.button("🚀 Generate Learning Plan", use_container_width=True):

        text_topic = None
        image_topic = None

        if user_query:
            text_topic = predict_topic(user_query)
            st.success(f"📌 NLP Detected Topic: **{text_topic}**")

        if uploaded_image:
            img = Image.open(uploaded_image)
            image_topic = predict_image(img)
            if image_topic is not None:
                st.info(f"🖼 Image Detected Topic: **{image_topic}**")
            else:
                st.warning(
                    "⚠️ **No relevant data found** — The uploaded image does not "
                    "appear to be related to any learning topic (e.g. ML, NLP, Deep "
                    "Learning, Web Development, etc.). Please upload a study-related image."
                )
                st.stop()  # Stop here — no recommendations for irrelevant images

        # Combine both signals
        final_topic = text_topic if text_topic else image_topic

        if final_topic:
            st.markdown(f'<span class="topic-chip">🏷 {final_topic}</span>', unsafe_allow_html=True)

            # ── Module 3: Learning Recommendation Model ──
            st.subheader("📚 Recommended Learning Resources")

            # Curated resources with filters
            resources = recommend(
                final_topic,
                level=level_filter if level_filter != "All" else None,
                sources=source_filter if source_filter else None,
            )

            # Real-time YouTube search
            videos = youtube_search(final_topic)

            has_any = (not resources.empty) or bool(videos)

            if not has_any:
                st.warning("No resources found. Try a different query.")
            else:
                # Curated resources
                if not resources.empty:
                    for _, row in resources.iterrows():
                        _render_resource_card(row, final_topic)

                # YouTube live results
                if videos:
                    st.markdown("#### 🎬 Real-Time YouTube Results")
                    for video in videos:
                        _render_youtube_card(video, final_topic)

            # ── Module 4: Quick Visualization ──
            if not resources.empty:
                st.divider()
                st.subheader("📊 Learning Path Visualization")

                v1, v2 = st.columns(2)
                with v1:
                    st.plotly_chart(
                        create_source_distribution_chart(resources),
                        use_container_width=True,
                    )
                with v2:
                    st.plotly_chart(
                        create_level_distribution_chart(resources),
                        use_container_width=True,
                    )

                st.plotly_chart(
                    create_learning_path_chart(resources, final_topic),
                    use_container_width=True,
                )
        else:
            st.warning("Please enter a query or upload an image to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Browse All Resources
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📚 Browse All Learning Resources")

    all_df = _load_resources()

    # Filters
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        browse_topic = st.selectbox("Topic", ["All"] + get_all_topics(), key="browse_topic")
    with bc2:
        browse_level = st.selectbox("Level", ["All", "Beginner", "Intermediate", "Advanced"], key="browse_level")
    with bc3:
        browse_source = st.selectbox("Platform", ["All"] + get_all_sources(), key="browse_source")

    filtered = all_df.copy()
    if browse_topic != "All":
        filtered = filtered[filtered["topic"] == browse_topic]
    if browse_level != "All":
        filtered = filtered[filtered["level"] == browse_level]
    if browse_source != "All":
        filtered = filtered[filtered["source"] == browse_source]

    # Stats
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-box"><h2>{len(filtered)}</h2><p>Resources Found</p></div>',
                    unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><h2>{filtered["source"].nunique()}</h2><p>Platforms</p></div>',
                    unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><h2>{filtered["topic"].nunique()}</h2><p>Topics</p></div>',
                    unsafe_allow_html=True)

    st.divider()

    for _, row in filtered.iterrows():
        _render_resource_card(row)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Education Visualization Dashboard (Module 4)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Education Visualization Dashboard")

    all_df = _load_resources()

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-box"><h2>{len(all_df)}</h2><p>Total Resources</p></div>',
                    unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><h2>{all_df["topic"].nunique()}</h2><p>Topics</p></div>',
                    unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><h2>{all_df["source"].nunique()}</h2><p>Platforms</p></div>',
                    unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><h2>{all_df["level"].nunique()}</h2><p>Levels</p></div>',
                    unsafe_allow_html=True)

    st.divider()

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_source_distribution_chart(all_df), use_container_width=True)
    with c2:
        st.plotly_chart(create_level_distribution_chart(all_df), use_container_width=True)

    st.divider()

    # Heatmap
    st.plotly_chart(create_topic_heatmap(all_df), use_container_width=True)

    st.divider()

    # Per-topic learning path
    viz_topic = st.selectbox("Select Topic for Learning Path", get_all_topics(), key="viz_topic")
    topic_resources = all_df[all_df["topic"] == viz_topic]
    st.plotly_chart(create_learning_path_chart(topic_resources, viz_topic), use_container_width=True)