import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
import pandas as pd  # <-- NUEVO: Añadir import de pandas
from sklearn.manifold import TSNE  # <-- NUEVO: Añadir import de TSNE

# --- NUEVO: DEFINICIÓN DE LA FUNCIÓN T-SNE ---
# Es buena práctica definir las funciones al principio del script.
def plot_tsne(pmf_model, params):
    print("\nGenerando visualización t-SNE de los factores latentes de películas...")
    # Cargar los nombres y géneros de las películas
    # Asegúrate de que el archivo u.item está en data/ml-100k/
    try:
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'u.item'. Asegúrate de que está en la carpeta 'data/ml-100k/'.")
        return

    # Extraer el género principal para cada película
    genre_cols = movies.columns[6:]
    movies['main_genre'] = movies[genre_cols].idxmax(axis=1)
    
    # Obtener la matriz de factores latentes de las películas que entrenamos
    item_factors = pmf_model.w_Item
    
    # Usar t-SNE para reducir la dimensionalidad a 2D
   # Línea corregida
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    item_factors_2d = tsne.fit_transform(item_factors)
    
    # Crear el DataFrame para el plot
    plot_df = pd.DataFrame(item_factors_2d, columns=['x', 'y'])
    # Nos aseguramos de que el movie_id empiece en 1 para hacer el join
    plot_df['movie_id'] = range(1, len(item_factors) + 1)
    plot_df = pd.merge(plot_df, movies[['movie_id', 'main_genre']], on='movie_id')
    
    # Crear el gráfico
    plt.figure(figsize=(16, 12))
    genres = plot_df['main_genre'].unique()
    for genre in genres:
        subset = plot_df[plot_df['main_genre'] == genre]
        plt.scatter(subset['x'], subset['y'], label=genre, alpha=0.7)
        
    plt.title('Visualización t-SNE de los Factores Latentes de Películas por Género')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1), markerscale=2)
    plt.grid(True)
    
    tsne_filename = f"tsne_E{params['maxepoch']}_F{params['num_feat']}.png"
    plt.tight_layout()
    plt.savefig(tsne_filename)
    print(f"Gráfica t-SNE guardada como: {tsne_filename}")


if __name__ == "__main__":
    # --- PARÁMETROS DEL EXPERIMENTO ---
    params = {
        "num_feat": 10,
        "epsilon": 1,
        "_lambda": 0.1,
        "momentum": 0.8,
        "maxepoch": 10,
        "num_batches": 100,
        "batch_size": 1000
    }
    # ----------------------------------

    print("Iniciando prueba con los siguientes parámetros:")
    print(params)

    file_path = "data/ml-100k/u.data"
    ratings = load_rating_data(file_path)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    pmf = PMF()
    pmf.set_params(params)
    pmf.fit(train, test)

    # --- GUARDAR GRÁFICAS Y MOSTRAR RESULTADOS ---
    
    # 1. Curva de Aprendizaje (RMSE)
    output_filename = f"curva_E{params['maxepoch']}_F{params['num_feat']}_L{params['_lambda']}.png"
    plt.figure() # Crea una nueva figura para evitar sobreescribir la anterior
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.savefig(output_filename)
    print(f"\nGráfica de la curva de aprendizaje guardada como: {output_filename}")

    # --- NUEVO: GRÁFICA DE DISTRIBUCIÓN DE PREDICCIONES ---
    print("\nGenerando gráfica de distribución de predicciones...")
    test_user_ids = np.array(test[:, 0], dtype='int32')
    test_item_ids = np.array(test[:, 1], dtype='int32')
    predictions = np.sum(np.multiply(pmf.w_User[test_user_ids, :], pmf.w_Item[test_item_ids, :]), axis=1) + pmf.mean_inv
    predictions = np.clip(predictions, 1, 5)
    real_ratings = test[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_ratings, bins=np.linspace(0.5, 5.5, 6), alpha=0.7, label='Ratings Reales', rwidth=0.8)
    plt.hist(predictions, bins=np.linspace(0.5, 5.5, 6), alpha=0.7, label='Predicciones del Modelo', rwidth=0.8)
    plt.title('Distribución de Ratings Reales vs. Predicciones')
    plt.xlabel('Rating')
    plt.ylabel('Frecuencia')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(axis='y')
    
    dist_filename = f"distribucion_E{params['maxepoch']}.png"
    plt.savefig(dist_filename)
    print(f"Gráfica de distribución guardada como: {dist_filename}")

    # --- NUEVO: LLAMADA A LA FUNCIÓN T-SNE ---
    plot_tsne(pmf, params)

    # --- RESULTADOS FINALES EN CONSOLA ---
    print("\nResultado final:")
    print(f"  Training RMSE: {pmf.rmse_train[-1]:.6f}")
    print(f"  Test RMSE:     {pmf.rmse_test[-1]:.6f}")
    print(f"  Precision, Recall: {pmf.topK(test)}")