# Federated Learning Simulator (FL Simulator)

An interactive, educational machine learning simulator designed to demonstrate how neural networks are trained collaboratively across edge clients without sharing private raw datasets. The simulator features custom partitioning skews, federated aggregation strategies, and privacy guarantees.

## 🚀 Key Features

*   **Client Data Partitioning**:
    *   **IID Split**: Data is shuffled and distributed equally across clients.
    *   **Dirichlet Non-IID Skew**: Simulates realistic client heterogeneity using a Dirichlet distribution $\operatorname{Dir}(\alpha)$ to create label concentration imbalances.
*   **Federated Aggregation Algorithms**:
    *   **FedAvg (McMahan et al.)**: Standard weighted average of updates based on client sample sizes.
    *   **FedProx (Li et al.)**: Integrates a proximal regularization term to stabilize convergence under high data skews.
    *   **FedAvgM**: Applies server-side momentum to speed up global convergence.
*   **Local Differential Privacy (DP)**: Simulates local DP noise injection on client weight updates (deltas) before aggregation. Shows the privacy-utility trade-off curves and $\epsilon$-budgets.
*   **2D Weight-Space Projection (PCA)**: Projects clients' neural network weights into 2D space, animating how local models drift apart during local updates and snap back together upon aggregation.
*   **Visualizations**: Stacked client label charts, interactive convergence curves, and PCA plots.
*   **Educational Math Explorer**: Deep-dives into Bellman-like equations, server aggregations, and DP noise formulations.

---

## 📁 File Structure

*   `app.py`: Main Streamlit interface styled with custom CSS glassmorphism.
*   `models.py`: Lightweight Multi-Layer Perceptron (MLP) classifier and serialization helpers.
*   `data_utils.py`: MNIST downloader and IID/Dirichlet partitioning split routines.
*   `client.py`: Local client training threads implementing FedProx regularization.
*   `server.py`: Central parameter server coordinating weight aggregations and global testing.
*   `viz_utils.py`: Interactive Plotly data distribution bar charts and PCA trajectory plots.
*   `test_fl.py`: Automated unit tests for client training and server aggregators.
*   `requirements.txt`: Project dependencies.

---

## 🛠️ Installation & Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run unit tests**:
    ```bash
    python -m unittest test_fl.py
    ```

3.  **Start the simulator console**:
    ```bash
    streamlit run app.py
    ```

---

## 🧮 Federated Mathematics

### FedAvg Update
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$

### FedProx Regularized Objective
$$\mathcal{L}_{\text{prox}}(w) = \mathcal{L}(w) + \frac{\mu}{2} \| w - w_t \|^2$$

### Differential Privacy Updates
$$\Delta_{\text{noisy}} = \Delta + \mathcal{N}\left(0, \sigma^2 S^2 \mathbf{I}\right)$$
