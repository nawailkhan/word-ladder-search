import streamlit as st
import time
from search_algorithms import (
    load_embeddings,
    bfs,
    dfs,
    ucs,
    greedy_best_first,
    a_star
)

# Page Configuration

st.set_page_config(
    page_title="Semantic Word Ladder AI",
    page_icon="🧠",
    layout="wide"
)

# Title Section
st.title("Semantic Word Ladder Search")
st.markdown("Explore search algorithms in high-dimensional embedding space.")

# Load Embeddings (cached)
@st.cache_resource
def load_data():
    return load_embeddings("glove.100d.20000.txt")

with st.spinner("Loading embeddings..."):
    embeddings = load_data()

# Sidebar Controls
st.sidebar.header("⚙️ Search Configuration")

start_word = st.sidebar.text_input("Start Word")
goal_word = st.sidebar.text_input("Goal Word")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["BFS", "DFS", "UCS", "Greedy Best-First", "A*"]
)

k = st.sidebar.slider("Number of Neighbors (k)", 5, 50, 10)

depth_limit = None
if algorithm == "DFS":
    depth_limit = st.sidebar.slider("Depth Limit", 5, 30, 15)

run_button = st.sidebar.button("Run Search")

# Execution
if run_button:
    if start_word not in embeddings:
        st.error(f"Start word '{start_word}' not in vocabulary.")
    elif goal_word not in embeddings:
        st.error(f"Goal word '{goal_word}' not in vocabulary.")
    else:
        st.info("Running search algorithm...")
        
        start_time = time.time()

        if algorithm == "BFS":
            result = bfs(start_word, goal_word, k, embeddings)
        elif algorithm == "DFS":
            result = dfs(start_word, goal_word, k, depth_limit, embeddings)
        elif algorithm == "UCS":
            result = ucs(start_word, goal_word, k, embeddings)
        elif algorithm == "Greedy Best-First":
            result = greedy_best_first(start_word, goal_word, k, embeddings)
        elif algorithm == "A*":
            result = a_star(start_word, goal_word, k, embeddings)

        path, length, nodes_expanded, runtime = result

    
        # Results Display
    
        st.subheader("📊 Results")

        col1, col2, col3 = st.columns(3)

        if path:
            col1.metric("Path Found", "✅ Yes")
            col2.metric("Path Length", length)
            col3.metric("Nodes Expanded", nodes_expanded)
        else:
            col1.metric("Path Found", "❌ No")
            col2.metric("Path Length", 0)
            col3.metric("Nodes Expanded", nodes_expanded)

        st.metric("Runtime (seconds)", f"{runtime:.4f}")

        if path:
            with st.expander("🔍 View Word Ladder"):
                st.write(" ➜ ".join(path))