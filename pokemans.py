import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the Pokemon dataset
def load_data(url):
    # Download the dataset using pandas
    df = pd.read_csv("Pokemon.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    # Select relevant statistics
    df = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

    # Drop rows with missing values
    df = df.dropna()

    return df

# Perform K-Means clustering
def perform_clustering(df, n_clusters):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df)
    return kmeans

# Analyze clusters for football positions
def analyze_clusters(kmeans, df):
    # Get cluster labels
    labels = kmeans.labels_

    # Add cluster labels to the DataFrame
    df["cluster"] = labels

    # Calculate mean statistics for each cluster
    cluster_means = df.groupby("cluster").mean()

    # Assign clusters to football positions (This is a simplified example)
    # In a real-world scenario, this would be based on a more sophisticated analysis
    position_map = {
        0: "Forward",
        1: "Midfielder",
        2: "Defender",
        3: "Goalkeeper",
    }

    # Assign clusters to football positions based on cluster means
    # This is a more sophisticated analysis than the previous example
    cluster_means = df.groupby("cluster").mean()

    # Define scoring metrics for each position
    def forward_score(row):
        return (row["Attack"] * 0.6) + (row["Speed"] * 0.4)

    def midfielder_score(row):
        return (row["Sp. Atk"] * 0.5) + (row["Sp. Def"] * 0.5)

    def defender_score(row):
        return (row["Defense"] * 0.7) + (row["HP"] * 0.3)

    def goalkeeper_score(row):
        return (row["Sp. Def"] * 0.8) + (row["HP"] * 0.2)

    # Assign positions based on the highest score
    df["forward_score"] = df.apply(forward_score, axis=1)
    df["midfielder_score"] = df.apply(midfielder_score, axis=1)
    df["defender_score"] = df.apply(defender_score, axis=1)
    df["goalkeeper_score"] = df.apply(goalkeeper_score, axis=1)

    def assign_position(row):
        scores = {
            "Forward": row["forward_score"],
            "Midfielder": row["midfielder_score"],
            "Defender": row["defender_score"],
            "Goalkeeper": row["goalkeeper_score"],
        }
        return max(scores, key=scores.get)

    df["position"] = df.apply(assign_position, axis=1)

    return df

# Create the Streamlit UI
def create_ui(df):
    st.title("Pokemon Football Team Selector")

    # Main screen with clickable football positions
    position = st.selectbox("Select a Football Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])

    # Filter the DataFrame based on the selected position
    position_df = df[df["position"] == position]

    # Position-specific detail view with three vertical sections
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Top 10 Pokemon")
        # Display the top 10 Pokemon for the selected position
        if position == "Forward":
            top_10 = position_df.sort_values("forward_score", ascending=False).head(10)
        elif position == "Midfielder":
            top_10 = position_df.sort_values("midfielder_score", ascending=False).head(10)
        elif position == "Defender":
            top_10 = position_df.sort_values("defender_score", ascending=False).head(10)
        else:
            top_10 = position_df.sort_values("goalkeeper_score", ascending=False).head(10)
        st.dataframe(top_10)

    with col2:
        st.header("Gradient Search Visualization")
        # Implement gradient search visualization
        if not position_df.empty:
            fig, ax = plt.subplots()
            ax.bar([1], [position_df["Attack"].mean()])
            st.pyplot(fig)
        else:
            st.write("No data available for this position.")

    with col3:
        st.header("Points and Lines Visualization")
        # Implement points and lines visualization
        fig, ax = plt.subplots()
        if position == "Forward":
            ax.scatter(position_df["Attack"], position_df["Speed"])
        elif position == "Midfielder":
            ax.scatter(position_df["Sp. Atk"], position_df["Sp. Def"])
        elif position == "Defender":
            ax.scatter(position_df["Defense"], position_df["HP"])
        else:
            ax.scatter(position_df["Sp. Def"], position_df["HP"])
        st.pyplot(fig)

if __name__ == "__main__":
    # Set the URL for the Pokemon dataset
    url = "https://www.kaggle.com/datasets/abcsds/pokemon"

    # Load the data
    df = load_data(url)

    # Preprocess the data
    df = preprocess_data(df)

    # Perform K-Means clustering
    n_clusters = 4  # Number of football positions
    kmeans = perform_clustering(df, n_clusters)

    # Analyze clusters
    df = analyze_clusters(kmeans, df)

    # Create the UI
    create_ui(df)
