import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Cargar el conjunto de datos de Pokemon
def load_data():
    # Descargar el conjunto de datos usando pandas
    df = pd.read_csv("Pokemon.csv")
    return df

# Preprocesar los datos
def preprocess_data(df):
    # Seleccionar estadísticas relevantes
    df = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

    # Eliminar filas con valores faltantes
    df = df.dropna()

    return df

# Realizar el clustering de K-Means
def perform_clustering(df, n_clusters):
    # Realizar el clustering de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(df)
    return kmeans

# Analizar los clusters para las posiciones de fútbol
def analyze_clusters(kmeans, df):
    # Obtener las etiquetas de los clusters
    labels = kmeans.labels_

    # Agregar las etiquetas de los clusters al DataFrame
    df["cluster"] = labels

    # Calcular las estadísticas medias para cada cluster
    cluster_means = df.groupby("cluster").mean()

    # Asignar clusters a posiciones de fútbol (Este es un ejemplo simplificado)
    # En un escenario del mundo real, esto se basaría en un análisis más sofisticado
    position_map = {
        0: "Forward",
        1: "Midfielder",
        2: "Defender",
        3: "Goalkeeper",
    }

    # Asignar clusters a posiciones de fútbol basándose en las medias de los clusters
    # Este es un análisis más sofisticado que el ejemplo anterior
    cluster_means = df.groupby("cluster").mean()

    # Definir las métricas de puntuación para cada posición
    def forward_score(row):
        return (row["Attack"] * 0.6) + (row["Speed"] * 0.4)

    def midfielder_score(row):
        return (row["Sp. Atk"] * 0.5) + (row["Sp. Def"] * 0.5)

    def defender_score(row):
        return (row["Defense"] * 0.7) + (row["HP"] * 0.3)

    def goalkeeper_score(row):
        return (row["Sp. Def"] * 0.8) + (row["HP"] * 0.2)

    # Asignar posiciones basándose en la puntuación más alta
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

# Crear la interfaz de usuario de Streamlit
def create_ui(df):
    st.title("Selector de Equipo de Fútbol de Pokemon")

    # Pantalla principal con posiciones de fútbol clickeables
    position = st.selectbox("Selecciona una Posición de Fútbol", ["Delantero", "Centrocampista", "Defensa", "Portero"], format_func=lambda x: {"Delantero": "Forward", "Centrocampista": "Midfielder", "Defensa": "Defender", "Portero": "Goalkeeper"}.get(x, x))

    # Map the selected position to the correct position name
    position_map = {
        "Delantero": "Forward",
        "Centrocampista": "Midfielder",
        "Defensa": "Defender",
        "Portero": "Goalkeeper",
    }
    selected_position = position_map[position]

    # Filter the DataFrame based on the selected position
    position_df = df[df["position"] == selected_position]

    # Position-specific detail view with three vertical sections
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Top 10 Pokemon")
        st.subheader("Los 10 mejores Pokemon")
        # Mostrar los 10 mejores Pokemon para la posición seleccionada
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
        st.header("Visualización de Búsqueda de Gradiente")
        # Implementar la visualización de búsqueda de gradiente
        if not position_df.empty:
            # Simulate gradient descent (simplified)
            attack_mean = position_df["Attack"].mean()
            attack_std = position_df["Attack"].std()
            x = np.linspace(attack_mean - 3 * attack_std, attack_mean + 3 * attack_std, 100)
            y = (x - attack_mean)**2  # Example loss function
            
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.scatter(attack_mean, 0, color="red")  # Mark the minimum
            ax.set_xlabel("Attack")
            ax.set_ylabel("Loss")
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para esta posición.")

    with col3:
        st.header("Visualización de Puntos y Líneas")
        # Implementar la visualización de puntos y líneas
        if not position_df.empty:
            # Scatter plot based on position
            fig, ax = plt.subplots()
            if selected_position == "Goalkeeper":
                x_data = position_df["Sp. Def"]
                y_data = position_df["Speed"]
                x_label = "Sp. Def"
                y_label = "Speed"
            elif selected_position == "Defender":
                x_data = position_df["Defense"]
                y_data = position_df["HP"]
                x_label = "Defense"
                y_label = "HP"
            elif selected_position == "Midfielder":
                x_data = position_df["Sp. Atk"]
                y_data = position_df["Speed"]
                x_label = "Sp. Atk"
                y_label = "Speed"
            else:  # Forward
                x_data = position_df["Attack"]
                y_data = position_df["Speed"]
                x_label = "Attack"
                y_label = "Speed"

            ax.scatter(x_data, y_data)

            # Add a regression line (simplified)
            m, b = np.polyfit(x_data, y_data, 1)
            x = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x, m*x + b, color="red")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para esta posición.")

if __name__ == "__main__":
    # Carga el csv de Pokemon
    df = load_data()

    # Preprocesar los datos
    df = preprocess_data(df)

    # Realizar el clustering de K-Means
    n_clusters = 4  # numero de posiciones de fútbol
    kmeans = perform_clustering(df, n_clusters)

    # Analizar los clusters
    df = analyze_clusters(kmeans, df)

    # Crear la UI
    create_ui(df)
