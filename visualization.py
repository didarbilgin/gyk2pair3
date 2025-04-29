import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class Visualization:
    def __init__(self, df):
        self.df = df

    def show_optimal_eps(self, distances, optimal_eps, kneedle, min_samples):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(distances, label=f'{min_samples}-th NN Distances')
            if kneedle.elbow is not None:
                plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'{min_samples}-th nearest neighbor distance')
            plt.title('Elbow Method for Optimal eps')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("outputs/visualization_plot.png")
            plt.close()

        except Exception as e:
            print(f"[ERROR] EPS grafiği gösterilemedi: {e}")

    def plot_clusters(self, x_col, y_col, cluster_col='cluster', title='DBSCAN Clustering'):
        """
        İki boyutlu kümeleme çıktısını çizmek için genel görselleştirme fonksiyonu.
        """
        if not all(col in self.df.columns for col in [x_col, y_col, cluster_col]):
            print(f"[WARNING] Plot için gerekli sütun(lar) bulunamadı: {x_col}, {y_col}, {cluster_col}")
            return

        try:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                self.df[x_col],
                self.df[y_col],
                c=self.df[cluster_col],
                cmap='plasma',
                s=60
            )
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(title)
            plt.grid(True)
            plt.colorbar(scatter, label='Küme No')
            plt.tight_layout()
            plt.savefig("outputs/visualization_cluster_plot.png")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Kümeleme görselleştirmesi başarısız: {e}")

    def print_outliers(self, id_column='product_id', cluster_col='cluster'):
        """
        Aykırı gözlemleri yazdırır (cluster = -1 olanlar)
        """
        try:
            if cluster_col not in self.df.columns:
                print(f"[WARNING] Aykırı verileri bulmak için '{cluster_col}' sütunu bulunamadı.")
                return

            outliers = self.df[self.df[cluster_col] == -1]
            print(f"Aykırı veri sayısı: {len(outliers)}")

            if id_column in outliers.columns:
                print(outliers[[id_column] + [col for col in outliers.columns if col not in [id_column, cluster_col]]])
            else:
                print(outliers.head())
        except Exception as e:
            print(f"[ERROR] Aykırı veriler yazdırılamadı: {e}")
