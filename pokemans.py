import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_data():
    # Descargar el conjunto de datos usando pandas
    df = pd.read_csv("Pokemon.csv")
    return df

# Preprocesar los datos
def preprocess_data(df):
    # Seleccionar estadísticas relevantes y mantener ID y Nombre
    df = df[["#", "Name", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

    # Eliminar filas con valores faltantes
    df = df.dropna()

    return df

# Realizar el clustering de K-Means
def perform_clustering(df, n_clusters):
    # Seleccionar solo las columnas numéricas para el clustering
    numerical_df = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    # Realizar el clustering de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(numerical_df)
    return kmeans

# Analizar los clusters para las posiciones de fútbol
def analyze_clusters(kmeans, df):
    # Obtener las etiquetas de los clusters
    labels = kmeans.labels_

    # Agregar las etiquetas de los clusters al DataFrame
    df["cluster"] = labels

    # Seleccionar solo las columnas numéricas para calcular las estadísticas medias
    numerical_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    cluster_means = df.groupby("cluster")[numerical_cols].mean()

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
def create_ui(df, pokemon_names):
    st.title("Selector de Equipo de Fútbol de Pokemon")

    # Pantalla principal con posiciones de fútbol clickeables
    position_options = ["Delantero", "Centrocampista", "Defensa", "Portero"]
    position_mapping = {
        "Delantero": "Forward",
        "Centrocampista": "Midfielder",
        "Defensa": "Defender",
        "Portero": "Goalkeeper",
    }
    position = st.selectbox(
        "Selecciona una Posición de Fútbol",
        position_options,
        format_func=lambda x: position_mapping.get(x, x),
    )

    # Map the selected position to the correct position name
    selected_position = position_mapping[position]

    # Filter the DataFrame based on the selected position
    position_df = df[df["position"] == selected_position].copy()

    # Position-specific detail view with two vertical sections
    col1, col2 = st.columns(2)

    with col1:
        st.header("Top 10 Pokemon")
        st.subheader("Los 10 mejores Pokemon")
        # Mostrar los 10 mejores Pokemon para la posición seleccionada
        sort_columns = {
            "Forward": "forward_score",
            "Midfielder": "midfielder_score",
            "Defender": "defender_score",
            "Goalkeeper": "goalkeeper_score",
        }
        sort_column = sort_columns[selected_position]
        top_10 = position_df.sort_values(sort_column, ascending=False).head(10)

        # Add Pokemon name to the data
        top_10 = top_10.reset_index(drop=True)
        st.dataframe(top_10[["#", "Name"] + [col for col in top_10.columns if col not in ["#", "Name", "cluster"]]])

    with col2:
        st.header("Visualización de Puntos y Líneas")
        # Implementar la visualización de puntos y líneas
        if not position_df.empty:
            # Scatter plot based on position
            fig, ax = plt.subplots()
            scatter_columns = {
                "Goalkeeper": {"x": "Sp. Def", "y": "Speed"},
                "Defender": {"x": "Defense", "y": "HP"},
                "Midfielder": {"x": "Sp. Atk", "y": "Speed"},
                "Forward": {"x": "Attack", "y": "Speed"},
            }
            x_data = position_df[scatter_columns[selected_position]["x"]]
            y_data = position_df[scatter_columns[selected_position]["y"]]
            x_label = scatter_columns[selected_position]["x"]
            y_label = scatter_columns[selected_position]["y"]

            ax.scatter(x_data, y_data)

            # Add a regression line (simplified)
            m, b = np.polyfit(x_data, y_data, 1)
            x = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x, m * x + b, color="red")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para esta posición.")

    # Soccer field visualization
    st.header("Equipo Ideal")
    st.subheader("Mejores Pokemones por Posición")

    # Define player positions and their corresponding Pokemon stats
    positions = {
        "Portero": "Goalkeeper",
        "Defensor": "Defender",
        "Mediocampista": "Midfielder",
        "Delantero": "Forward",
    }

    # Display the best Pokemon for each position
    for pos_name, pos_value in positions.items():
        if pos_value == "Goalkeeper":
            best_pokemon = (
                df[df["position"] == pos_value]
                .sort_values("goalkeeper_score", ascending=False)
                .iloc[0]
            )
        elif pos_value == "Defender":
            best_pokemon = (
                df[df["position"] == pos_value]
                .sort_values("defender_score", ascending=False)
                .iloc[0]
            )
        elif pos_value == "Midfielder":
            best_pokemon = (
                df[df["position"] == pos_value]
                .sort_values("midfielder_score", ascending=False)
                .iloc[0]
            )
        else:
            best_pokemon = (
                df[df["position"] == pos_value]
                .sort_values("forward_score", ascending=False)
                .iloc[0]
            )

        st.write(
            f"**{pos_name}:** {pokemon_names[best_pokemon['#']]} (ID: {best_pokemon['#']})"
        )

if __name__ == "__main__":
    # Carga el csv de Pokemon
    pokemon_df = pd.read_csv("Pokemon.csv")
    pokemon_names = dict(zip(pokemon_df["#"], pokemon_df["Name"]))

    df = load_data()

    # Preprocesar los datos
    df = preprocess_data(df)

    # Realizar el clustering de K-Means
    n_clusters = 4  # numero de posiciones de fútbol
    kmeans = perform_clustering(df, n_clusters)

    # Analizar los clusters
    df = analyze_clusters(kmeans, df)

    # Crear la UI
    create_ui(df, pokemon_names)
