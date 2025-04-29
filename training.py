import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import visualization
import matplotlib
matplotlib.use("Agg")  # GUI backend yerine dosya çıktısı alınmasını sağlar


class Training:
    def __init__(self, df):
        self.df = df

    def find_optimal_eps(self, X_scaled, min_samples):
        """
        K-distance grafiğiyle optimal eps belirleme.
        """
        try:
            # K-nearest neighbors ile mesafeleri hesapla
            neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
            distances, _ = neighbors.kneighbors(X_scaled)
            distances = np.sort(distances[:, -1] + 1e-10)  # Küçük epsilon ekleyerek sıfır bölmeyi engelle

            # Elbow metodunu kullanarak optimal eps belirle
            kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
            optimal_eps = distances[kneedle.elbow] if kneedle.elbow is not None else distances[-1]  # Son mesafe değeri

            # EPS grafiğini göster
            visualization.Visualization(self.df).show_optimal_eps(distances, optimal_eps, kneedle, min_samples)

            return optimal_eps
        except ValueError as e:
            print(f"[ERROR] Değer hatası: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Optimal eps hesaplanamadı: {e}")
            return None
    def optimize_dbscan_params(self, X_scaled, eps_values=np.arange(0.1, 2.0, 0.1), min_samples_list=[3, 5, 10, 20]):
        """
        Silhouette skoru kullanarak en iyi eps ve min_samples parametrelerini bulur.
        """
        best_score = -1
        best_params = {'eps': None, 'min_samples': None}
        scores = []

        for min_samples in min_samples_list:
            for eps in eps_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters <= 1:
                        continue

                    score = silhouette_score(X_scaled, labels)
                    scores.append((eps, min_samples, score))

                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                except Exception:
                    continue

        # Skorları çiz
        if scores:
            try:
                eps_list, min_samples_list_plot, silhouette_scores = zip(*scores)
                plt.figure(figsize=(12, 6))
                scatter = plt.scatter(eps_list, min_samples_list_plot, c=silhouette_scores, cmap='viridis', s=100)
                plt.colorbar(scatter, label='Silhouette Score')
                plt.xlabel('eps')
                plt.ylabel('min_samples')
                plt.title('DBSCAN Parametre Optimizasyonu (Silhouette Score)')
                plt.grid(True)
                plt.savefig("outputs/training_plot.png")  # veya uygun bir isim
                plt.close()

            except Exception as e:
                print(f"[WARNING] Silhouette scatter çizimi başarısız: {e}")

        return best_params['eps'], best_params['min_samples']

    def train_model(self, feature_columns):
        """
        Verilen feature sütunlarına göre DBSCAN eğitimi yapar.
        """
        try:
            X = self.df[feature_columns].values
        except KeyError as e:
            raise ValueError(f"Girdi sütunları dataframe'de eksik: {e}")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Parametre optimizasyonu
        optimized_eps, optimized_min_samples = self.optimize_dbscan_params(X_scaled)

        if optimized_eps is None or optimized_min_samples is None:
            print("Uygun parametre bulunamadı, default ayarlarla devam ediliyor.")
            optimized_eps = 0.5
            optimized_min_samples = 5

        print(f"[INFO] Optimize eps: {optimized_eps}, min_samples: {optimized_min_samples}")

        # K-distance grafiğiyle eps doğrulama
        k_distance_eps = self.find_optimal_eps(X_scaled, optimized_min_samples)
        print(f"[INFO] K-distance ile önerilen eps: {k_distance_eps}")

        # DBSCAN eğitimi
        dbscan = DBSCAN(eps=optimized_eps, min_samples=optimized_min_samples)
        self.df['cluster'] = dbscan.fit_predict(X_scaled)

        print(f"[INFO] Kümeleme tamamlandı. Toplam küme sayısı: {len(set(self.df['cluster'])) - (1 if -1 in self.df['cluster'] else 0)}")

        return self.df, dbscan
