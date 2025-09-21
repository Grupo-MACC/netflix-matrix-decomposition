import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from LoadData import load_rating_data
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

# ==========================================================================================
# --- ANÁLISIS 1: FUNCIÓN PARA EL "MAPA DE GUSTOS" DE USUARIOS (t-SNE de Usuarios) ---
# ==========================================================================================
def plot_user_tsne(pmf_model, params):
    print("\n--- ANÁLISIS 1: Generando 'Mapa de Gustos' de Usuarios por Profesión (t-SNE) ---")
    try:
        # Cargar los datos demográficos de los usuarios desde u.user
        u_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'u.user'. Asegúrate de que está en 'data/ml-100k/'.")
        return

    # Extraer la matriz de factores latentes de los usuarios del modelo entrenado
    user_factors = pmf_model.w_User
    
    # Aplicar t-SNE para reducir las N dimensiones de los factores a solo 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    user_factors_2d = tsne.fit_transform(user_factors)

    # Preparar los datos para el gráfico
    plot_df = pd.DataFrame(user_factors_2d, columns=['x', 'y'])
    # El ID de usuario en la matriz empieza en 0, pero en el fichero en 1. Ajustamos.
    plot_df['user_id'] = range(0, len(user_factors))
    plot_df = pd.merge(plot_df, users, on='user_id')

    # Crear el gráfico
    plt.figure(figsize=(16, 12))
    occupations = plot_df['occupation'].unique()
    for occ in occupations:
        subset = plot_df[plot_df['occupation'] == occ]
        plt.scatter(subset['x'], subset['y'], label=occ, alpha=0.7)
    
    plt.title('Visualización t-SNE de los Factores Latentes de Usuarios por Profesión')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1), markerscale=2)
    plt.grid(True)
    
    tsne_filename = f"tsne_usuarios_E{params['maxepoch']}.png"
    plt.tight_layout()
    plt.savefig(tsne_filename)
    print(f"Gráfica t-SNE de usuarios guardada como: {tsne_filename}")

# ==========================================================================================
# --- ANÁLISIS 2: FUNCIÓN PARA EL ESTUDIO DE CASO DE UN USUARIO ESPECÍFICO ---
# ==========================================================================================
def analyze_single_user(pmf_model, user_id, ratings_df, movies_df):
    print(f"\n--- ANÁLISIS 2: Estudio de Caso para el Usuario #{user_id} ---")
    
    # --- Historial del Usuario ---
    user_history = ratings_df[ratings_df['user_id'] == user_id].sort_values('rating', ascending=False)
    user_history = pd.merge(user_history, movies_df[['movie_id', 'title']], on='movie_id')
    
    print(f"\nLas 5 películas que más le gustaron al Usuario #{user_id}:")
    print(user_history.head(5)[['title', 'rating']].to_string(index=False))

    print(f"\nLas 5 películas que menos le gustaron al Usuario #{user_id}:")
    print(user_history.tail(5).sort_values('rating')[['title', 'rating']].to_string(index=False))

    # --- Recomendaciones del Modelo ---
    # Obtener las predicciones para TODAS las películas para este usuario
    all_predictions = pmf_model.predict(user_id)
    predictions_df = pd.DataFrame({
        'movie_id': range(1, len(all_predictions) + 1),
        'predicted_rating': all_predictions
    })

    # Filtrar las películas que el usuario ya ha visto
    movies_seen = user_history['movie_id'].tolist()
    recommendations = predictions_df[~predictions_df['movie_id'].isin(movies_seen)]
    
    # Ordenar por la predicción más alta
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    # Unir con los títulos de las películas
    recommendations = pd.merge(recommendations, movies_df[['movie_id', 'title']], on='movie_id')
    
    print(f"\nTOP 10 recomendaciones del modelo para el Usuario #{user_id}:")
    print(recommendations.head(10)[['title', 'predicted_rating']].to_string(index=False))

# ==========================================================================================
# --- ANÁLISIS 3: FUNCIÓN PARA LA "AUTOPSIA DEL MODELO" (PEORES ERRORES) ---
# ==========================================================================================
def analyze_worst_errors(pmf_model, test_data, movies_df):
    print("\n--- ANÁLISIS 3: 'Autopsia del Modelo' - Los 10 Peores Errores de Predicción ---")
    
    # Calcular predicciones para todo el conjunto de prueba
    test_user_ids = np.array(test_data[:, 0], dtype='int32')
    test_item_ids = np.array(test_data[:, 1], dtype='int32')
    predictions = np.sum(np.multiply(pmf_model.w_User[test_user_ids, :], pmf_model.w_Item[test_item_ids, :]), axis=1) + pmf_model.mean_inv

    # Crear un DataFrame con los resultados
    errors_df = pd.DataFrame({
        'user_id': test_data[:, 0],
        'movie_id': test_data[:, 1],
        'real_rating': test_data[:, 2],
        'predicted_rating': predictions
    })
    
    # Calcular el error absoluto
    errors_df['error'] = (errors_df['real_rating'] - errors_df['predicted_rating']).abs()
    
    # Ordenar por el error más grande
    worst_errors = errors_df.sort_values('error', ascending=False)
    
    # Unir con los títulos de las películas para que sea legible
    worst_errors = pd.merge(worst_errors, movies_df[['movie_id', 'title']], on='movie_id')
    
    print("\nLas predicciones con mayor diferencia entre el rating real y el predicho:")
    print(worst_errors.head(10)[['user_id', 'title', 'real_rating', 'predicted_rating', 'error']].to_string(index=False))


# ==========================================================================================
# --- SCRIPT PRINCIPAL ---
# ==========================================================================================
if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    params = {
        "num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8,
        "maxepoch": 15, "num_batches": 100, "batch_size": 1000
    }
    # Elige un ID de usuario para el Análisis 2. El usuario 196 tiene muchos ratings.
    USER_ID_FOR_ANALYSIS = 196
    
    print("Iniciando el entrenamiento y análisis avanzado del modelo PMF...")
    print("Parámetros:", params)

    # --- CARGA DE DATOS ---
    ratings = load_rating_data("data/ml-100k/u.data")
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Cargar datos de películas para usarlos en los análisis
    movie_title_cols = ['movie_id', 'title']
    movies_df = pd.read_csv('data/ml-100k/u.item', sep='|', names=movie_title_cols, usecols=[0, 1], encoding='latin-1')

    # Cargar datos de ratings en un DataFrame para el análisis de usuario
    # --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'])

    # --- ENTRENAMIENTO ---
    pmf = PMF()
    pmf.set_params(params)
    pmf.fit(train, test)
    print("\nEntrenamiento del modelo completado.")
    print(f"Resultado final -> Test RMSE: {pmf.rmse_test[-1]:.6f}, Precision/Recall: {pmf.topK(test)}")

    # --- EJECUCIÓN DE LOS ANÁLISIS AVANZADOS ---
    # 1. Mapa de Gustos de Usuarios
    plot_user_tsne(pmf, params)
    
    # 2. Estudio de Caso de un Usuario
    analyze_single_user(pmf, USER_ID_FOR_ANALYSIS, ratings_df, movies_df)
    
    # 3. Autopsia del Modelo
    analyze_worst_errors(pmf, test, movies_df)
    
    print("\n\nAnálisis avanzado completado.")