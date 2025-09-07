Student Data Clustering Project

This repository explores clustering techniques on the student performance dataset.
The work includes preprocessing, visualization, clustering with DBSCAN, Hierarchical Clustering, and KMeans, and comparison of evaluation metrics.

Dataset

File: data_student_augmented.csv

Contains both categorical and numerical features.

Target column UNS is removed since clustering is unsupervised.

Preprocessing

Handling categorical features: Label encoding used for non-numeric columns.

Standardization: All features scaled using StandardScaler to ensure equal contribution.

Data cleaning: Removed irrelevant/target columns (UNS).

Data Visualization

Distribution plots to understand feature ranges.

Correlation heatmap to study relationships between variables.

PCA/2D scatter plots for visualizing cluster separation.

Clustering Methods
1. DBSCAN

Density-based clustering.

Parameters tuned: eps, min_samples.

Handles noise points (label -1).

2. Hierarchical (Agglomerative) Clustering

Linkage method: Ward.

Number of clusters defined based on dendrogram and silhouette analysis.

3. KMeans

Centroid-based clustering.

Optimal k determined using Elbow Method and Silhouette Analysis.

Evaluation Metrics

To compare clustering models, the following metrics were used:

Silhouette Score: Measures how well clusters are defined (higher is better).

Davies–Bouldin Index: Measures similarity between clusters (lower is better).

Calinski–Harabasz Index: Measures separation between clusters (higher is better).

Results

Each clustering method was evaluated and compared using the above metrics.

DBSCAN can detect noise but may struggle with high-dimensionality.

Hierarchical clustering provides interpretable dendrograms.

KMeans gave stable results with optimal k chosen via the elbow method.

How to Run

Clone this repository and install required packages:

pip install -r requirements.txt


Run clustering scripts:

python clustering_dbscan.py
python clustering_hierarchical.py
python clustering_kmeans.py

Next Steps

Apply dimensionality reduction (PCA/t-SNE) before clustering.

Experiment with other clustering models (Gaussian Mixture Models, OPTICS).

Automate evaluation metric comparison for all models.