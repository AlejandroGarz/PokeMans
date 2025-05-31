import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

def draw_soccer_field(best_pokemons_by_position):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)

    # Dibuja el campo de fútbol
    try:
        img = plt.imread("Cancha.png")
        ax.imshow(img, extent=[0, 100, 0, 70], aspect='auto')
    except FileNotFoundError:
        st.warning("Cancha.png no encontrada. Asegúrate de que la imagen esté en el mismo directorio.")
        ax.set_facecolor("green") 


    # Posiciones de los jugadores en el campo
    # Define las coordenadas de los jugadores en el campo
    player_coords = {
        "Portero": [(5, 35)],
        "Defensor": [(20, 15), (20, 30), (20, 40), (20, 55)],
        "Mediocampista": [(50, 20), (50, 35), (50, 50)],
        "Delantero": [(80, 20), (80, 35), (80, 50)],
    }

    # Dibuja los jugadores en el campo
    for position_type, coords_list in player_coords.items():
        pokemons_for_position = best_pokemons_by_position.get(position_type, [])
        for i, (x, y) in enumerate(coords_list):
            if i < len(pokemons_for_position):
                pokemon = pokemons_for_position[i]
                ax.text(x, y, f"{pokemon['Name']}\n(ID: {pokemon['#']})",
                        color="yellow", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))
            else:
                # Si no hay suficientes Pokemon, muestra un marcador genérico
                ax.text(x, y, f"{position_type} {i+1}",
                        color="gray", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))

    ax.axis("off") # Desactiva los ejes
    st.pyplot(fig)

def load_data():
    # Descarga el conjunto de datos usando pandas
    df = pd.read_csv("Pokemon.csv")
    return df

# Preprocesa los datos
def preprocess_data(df):
    # Selecciona estadísticas relevantes y mantiene ID y Nombre
    df = df[["#", "Name", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

    # Elimina filas con valores faltantes
    df = df.dropna()

    return df

# Realiza el clustering de K-Means
def perform_clustering(df, n_clusters):
    # Selecciona solo las columnas numéricas para el clustering
    numerical_df = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    # Realiza el clustering de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(numerical_df)
    return kmeans

# Analiza los clusters para las posiciones de fútbol
def analyze_clusters(kmeans, df):
    # Obtiene las etiquetas de los clusters
    labels = kmeans.labels_

    # Agrega las etiquetas de los clusters al DataFrame
    df["cluster"] = labels

    # Selecciona solo las columnas numéricas para calcular las estadísticas medias
    numerical_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    cluster_means = df.groupby("cluster")[numerical_cols].mean()

    # Asigna clusters a posiciones de fútbo
    position_map = {
        0: "Delantero",
        1: "Centrocampista",
        2: "Defensa",
        3: "Portero",
    }

    # Asigna clusters a posiciones de fútbol basándose en las medias de los clusters
    

    # Define las métricas de puntuación para cada posición
    def Delantero_score(row):
        return (row["Attack"] * 0.6) + (row["Speed"] * 0.4)

    def Mediocampista_score(row):
        return (row["Sp. Atk"] * 0.5) + (row["Sp. Def"] * 0.5)

    def Defensor_score(row):
        return (row["Defense"] * 0.7) + (row["HP"] * 0.3)

    def goalkeeper_score(row):
        return (row["Sp. Def"] * 0.8) + (row["HP"] * 0.2)

    # Asigna posiciones basándose en la puntuación más alta
    df["Delantero_score"] = df.apply(Delantero_score, axis=1)
    df["Mediocampista_score"] = df.apply(Mediocampista_score, axis=1)
    df["Defensor_score"] = df.apply(Defensor_score, axis=1)
    df["goalkeeper_score"] = df.apply(goalkeeper_score, axis=1)

    def assign_position(row):
        scores = {
            "Delantero": row["Delantero_score"],
            "Mediocampista": row["Mediocampista_score"],
            "Defensor": row["Defensor_score"],
            "Portero": row["goalkeeper_score"],
        }
        return max(scores, key=scores.get)

    df["position"] = df.apply(assign_position, axis=1)

    return df

# Crea la interfaz de usuario de Streamlit
def create_ui(df, pokemon_names, pokemon_df):
    st.title("Selector de Equipo de Fútbol de Pokemon")

    # Pantalla principal con posiciones de fútbol clickeables
    position_options = ["Delantero", "Centrocampista", "Defensa", "Portero"]
    position_mapping = {
        "Delantero": "Delantero",
        "Centrocampista": "Mediocampista",
        "Defensa": "Defensor",
        "Portero": "Portero",
    }
    position = st.selectbox(
        "Selecciona una Posición de Fútbol",
        position_options,
        format_func=lambda x: position_mapping.get(x, x),
    )

    # Mapea la posición seleccionada al nombre de posición correcto
    selected_position = position_mapping[position]

    # Filtra el DataFrame basándose en la posición seleccionada
    position_df = df[df["position"] == selected_position].copy()

    # Vista detallada específica de la posición con dos secciones verticales
    col1, col2 = st.columns(2)

    with col1:
        st.header("Top 10 Pokemon")
        st.subheader("Los 10 mejores Pokemon")
        # Muestra los 10 mejores Pokemon para la posición seleccionada
        sort_columns = {
            "Delantero": "Delantero_score",
            "Mediocampista": "Mediocampista_score",
            "Defensor": "Defensor_score",
            "Portero": "goalkeeper_score",
        }
        sort_column = sort_columns[selected_position]
        top_10 = position_df.sort_values(sort_column, ascending=False).head(10)

        # Agrega el nombre del Pokemon a los datos
        top_10 = top_10.reset_index(drop=True)
        st.dataframe(top_10[["#", "Name"] + [col for col in top_10.columns if col not in ["#", "Name", "cluster"]]])

    with col2:
        st.header("Visualización de Puntos y Líneas")
        # Implementa la visualización de puntos y líneas
        if not position_df.empty:
            # Gráfico de dispersión basado en la posición
            fig, ax = plt.subplots()
            scatter_columns = {
                "Portero": {"x": "Sp. Def", "y": "Speed"},
                "Defensor": {"x": "Defense", "y": "HP"},
                "Mediocampista": {"x": "Sp. Atk", "y": "Speed"},
                "Delantero": {"x": "Attack", "y": "Speed"},
            }
            x_data = position_df[scatter_columns[selected_position]["x"]]
            y_data = position_df[scatter_columns[selected_position]["y"]]
            x_label = scatter_columns[selected_position]["x"]
            y_label = scatter_columns[selected_position]["y"]

            ax.scatter(x_data, y_data)

            # Agrega una línea de regresión (simplificado)
            m, b = np.polyfit(x_data, y_data, 1)
            x = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x, m * x + b, color="red")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para esta posición.")

    # Visualización de la cancha de fútbol
    st.header("Equipo Ideal")
    st.subheader("Mejores Pokémon por Posición")

    # Define las posiciones de los jugadores y sus estadísticas de Pokemon correspondientes
    positions = {
        "Portero": "Portero",
        "Defensor": "Defensor",
        "Mediocampista": "Mediocampista",
        "Delantero": "Delantero",
    }

    best_pokemons_by_position = {
        "Portero": [],
        "Defensor": [],
        "Mediocampista": [],
        "Delantero": []
    }
    selected_pokemon_ids = set()

    # Define el número de jugadores para cada posición
    num_players = {
        "Portero": 1,
        "Defensor": 4,
        "Mediocampista": 3,
        "Delantero": 3
    }

    # Define las columnas de puntuación para cada posición
    score_columns = {
        "Portero": "goalkeeper_score",
        "Defensor": "Defensor_score",
        "Mediocampista": "Mediocampista_score",
        "Delantero": "Delantero_score",
    }

    # Itera a través de las posiciones y selecciona Pokemon únicos
    for pos_value in ["Portero", "Defensor", "Mediocampista", "Delantero"]:
        current_position_candidates = df[df["position"] == pos_value].copy()
        current_position_candidates = current_position_candidates.sort_values(
            score_columns[pos_value], ascending=False
        )

        count = 0
        for index, pokemon in current_position_candidates.iterrows():
            if pokemon["#"] not in selected_pokemon_ids:
                best_pokemons_by_position[pos_value].append(pokemon.to_dict())
                selected_pokemon_ids.add(pokemon["#"])
                count += 1
            if count >= num_players[pos_value]:
                break
    
    draw_soccer_field(best_pokemons_by_position)

if __name__ == "__main__":
    # Carga el csv de Pokemon
    pokemon_df = pd.read_csv("Pokemon.csv")
    pokemon_names = dict(zip(pokemon_df["#"], pokemon_df["Name"]))

    df = load_data()

    # Preprocesa los datos
    df = preprocess_data(df)

    # Realiza el clustering de K-Means
    n_clusters = 4  # numero de posiciones de fútbol
    kmeans = perform_clustering(df, n_clusters)

    # Analiza los clusters
    df = analyze_clusters(kmeans, df)

    # Crea la UI
    create_ui(df, pokemon_names, pokemon_df)
