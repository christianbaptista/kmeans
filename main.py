from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar dados genéricos
categories = ['comp.graphics', 'rec.sport.hockey', 'sci.med', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data

# Converter texto em vetores TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(texts)

# Determinar o número ideal de clusters (método do cotovelo)
def find_optimal_clusters(data, max_k):
    scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        scores.append(kmeans.inertia_)
    plt.plot(range(2, max_k), scores, marker='o')
    plt.title("Método do Cotovelo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inércia")
    plt.show()

find_optimal_clusters(X, 10)

# Treinar o modelo com o número de clusters ideal
num_clusters = 4  # Ajuste com base no gráfico
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Verificar os clusters
clusters = kmeans.labels_

# Mostrar exemplos de cada cluster
for cluster in range(num_clusters):
    print(f"\nCluster {cluster}:")
    for i, text in enumerate([texts[i] for i in range(len(clusters)) if clusters[i] == cluster][:5]):
        print(f"  {i+1}: {text[:200]}...")
