import gradio as gr
import joblib
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

USERS = {
    "user": "123",
}

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

X_pca = joblib.load("X_pca.pkl")
labels_dict_train = joblib.load("labels_dict.pkl")
hier_centroids = joblib.load("hier_centroids.pkl")
db_centroids = joblib.load("db_centroids.pkl")

def assign_by_nearest_centroid(x_scaled, centroids):
    best_lbl = None
    best_dist = float("inf")
    for lbl, c in centroids.items():
        d = np.linalg.norm(x_scaled.ravel() - c)
        if d < best_dist:
            best_dist = d
            best_lbl = int(lbl)
    return best_lbl

def create_combined_plot(X_pca, labels_dict, x_input_pca, pred_labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    algos = ['KMeans', 'Hierarchical', 'DBSCAN']
    cmap = plt.cm.get_cmap("tab10")
    for i, algo in enumerate(algos):
        ax = axes[i]
        lbls = labels_dict[algo]
        unique = np.unique(lbls)
        for j, u in enumerate(unique):
            mask = lbls == u
            color = cmap(j % 10) if u != -1 else (0.6,0.6,0.6)
            ax.scatter(X_pca[mask,0], X_pca[mask,1], c=[color], s=40, alpha=0.6, edgecolors='k', linewidths=0.2)
        ax.scatter(x_input_pca[0,0], x_input_pca[0,1], c='red', marker='X', s=200, edgecolors='k')
        ax.set_title(f"{algo} (pred: {pred_labels.get(algo,'N/A')})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def predict_and_visualize(*user_inputs):
    user_arr = np.array([user_inputs], dtype=float)
    user_scaled = scaler.transform(user_arr)
    km_label = int(kmeans.predict(user_scaled)[0])
    hier_label = assign_by_nearest_centroid(user_scaled, hier_centroids)
    db_label = assign_by_nearest_centroid(user_scaled, db_centroids)
    if db_label is None:
        db_label = -1
    user_pca = pca.transform(user_scaled)
    pred_labels = {'KMeans': km_label, 'Hierarchical': hier_label, 'DBSCAN': db_label}
    img = create_combined_plot(X_pca, labels_dict_train, user_pca, pred_labels)
    return f"{km_label}", f"{hier_label}", f"{db_label}", img

def check_login(username, password):
    return username in USERS and USERS[username] == password

with gr.Blocks(css=".login-card {max-width: 400px; margin: auto; padding: 20px; box-shadow: 0px 2px 8px rgba(0,0,0,0.1); border-radius: 10px;}") as demo:
    with gr.Row():
        with gr.Column(elem_classes="login-card") as login_ui:
            gr.Markdown("## 🔐 Welcome to Clustering Explorer")
            username_input = gr.Textbox(label="Username", placeholder="Enter username")
            password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            login_btn = gr.Button("Login", variant="primary")
            login_message = gr.HTML()

    with gr.Row(visible=False) as main_ui:
        gr.Markdown("# 🧮 Customer Clustering Explorer")
        input_fields = [gr.Number(label=col) for col in ["STG", "SCG", "STR", "LPR", "PEG"]]
        predict_btn = gr.Button("Predict Clusters")
        km_out = gr.Textbox(label="KMeans Cluster")
        hier_out = gr.Textbox(label="Hierarchical Cluster")
        db_out = gr.Textbox(label="DBSCAN Cluster")
        plot_out = gr.Image(label="Cluster visualization (PCA 2D)")
        predict_btn.click(fn=predict_and_visualize,
                          inputs=input_fields,
                          outputs=[km_out, hier_out, db_out, plot_out])

    def login_action(username, password):
        if check_login(username, password):
            return "<p style='color:green'>✅ Login successful! You can now use the clustering tool.</p>", gr.update(visible=True), gr.update(visible=False)
        else:
            return "<p style='color:red'>❌ Login failed! Please try again.</p>", gr.update(visible=False), gr.update(visible=True)

    login_btn.click(fn=login_action,
                    inputs=[username_input, password_input],
                    outputs=[login_message, main_ui, login_ui])

if __name__ == "__main__":
    demo.launch()

