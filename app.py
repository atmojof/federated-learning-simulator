import streamlit as st
import numpy as np
import pandas as pd
import torch
import copy
import time

# Import FL Simulator modules
from models import SimpleMLP, get_weights, set_weights
from data_utils import get_mnist_data, partition_data
from client import FLClient
from server import FLServer
from viz_utils import (
    plot_client_distributions, plot_weight_drift,
    plot_fl_metrics, plot_comparison_metrics
)

# Page Setup
st.set_page_config(
    page_title="Federated Learning Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for modern slate look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
}

.main-title {
    background: linear-gradient(135deg, #34d399 0%, #3b82f6 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.premium-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.metric-card {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(52, 211, 153, 0.3);
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #34d399;
    font-family: 'Space Grotesk', sans-serif;
}

.metric-lbl {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}

.math-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(52, 211, 153, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if 'train_data' not in st.session_state:
    with st.spinner("Downloading or loading dataset..."):
        train_x, train_y, test_x, test_y, source_name = get_mnist_data()
        st.session_state.train_x = train_x
        st.session_state.train_y = train_y
        st.session_state.test_x = test_x
        st.session_state.test_y = test_y
        st.session_state.dataset_source = source_name

if 'clients_data' not in st.session_state:
    st.session_state.clients_data = None
if 'active_clients' not in st.session_state:
    st.session_state.active_clients = []
if 'server' not in st.session_state:
    st.session_state.server = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'drift_snapshots' not in st.session_state:
    st.session_state.drift_snapshots = []
if 'runs_comparison' not in st.session_state:
    st.session_state.runs_comparison = {}

# Title
st.markdown('<div class="main-title">🤖 Federated Learning Simulator</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Distributed Deep Learning Playground &mdash; Data Source: <b>{st.session_state.dataset_source}</b></div>', unsafe_allow_html=True)

# ----------------- SIDEBAR: SYSTEM PARAMETERS -----------------
st.sidebar.markdown("### ⚙️ Simulator Setup")

num_clients = st.sidebar.slider("Number of Clients", min_value=3, max_value=10, value=5)

partition_type = st.sidebar.radio(
    "Data Partitioning Strategy",
    ["IID (Equal Shuffle)", "Non-IID (Dirichlet Skew)"]
)

if "Dirichlet" in partition_type:
    alpha = st.sidebar.slider(
        "Dirichlet Alpha (Skew Concentration)",
        min_value=0.05, max_value=2.0, value=0.2, step=0.05,
        help="Smaller values = more severe data skew (Non-IID). Larger values = more homogeneous."
    )
else:
    alpha = 1.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Neural Network Config")
hidden_dim = st.sidebar.selectbox("Hidden Layer Dimension", [32, 64, 128], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Local Optimization")
algorithm = st.sidebar.selectbox("Federated Algorithm", ["FedAvg", "FedProx", "FedAvgM"])

local_epochs = st.sidebar.slider("Local Client Epochs", min_value=1, max_value=5, value=2)
local_batch_size = st.sidebar.selectbox("Local Batch Size", [16, 32, 64], index=1)
local_lr = st.sidebar.number_input("Local Learning Rate (eta)", min_value=0.001, max_value=0.5, value=0.05, format="%.3f")

# Algorithm-specific hyperparameters
if algorithm == "FedProx":
    mu = st.sidebar.slider(
        "Proximal Penalty (mu)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Enforces that local weight updates don't drift too far from the global server model."
    )
else:
    mu = 0.0

if algorithm == "FedAvgM":
    momentum_beta = st.sidebar.slider("Server Momentum (beta)", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
else:
    momentum_beta = 0.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Privacy Controls")
dp_noise = st.sidebar.slider(
    "Local DP Noise Multiplier (sigma)",
    min_value=0.0, max_value=1.5, value=0.0, step=0.1,
    help="Adds calibrated Gaussian noise to the aggregated client updates to guarantee Differential Privacy."
)

# Trigger data partitioning when configuration changes
config_hash = f"{num_clients}_{partition_type}_{alpha}"
if 'prev_config_hash' not in st.session_state or st.session_state.prev_config_hash != config_hash:
    st.session_state.prev_config_hash = config_hash
    st.session_state.history = None # Reset simulation run
    st.session_state.drift_snapshots = []
    
    with st.spinner("Partitioning data among clients..."):
        p_type = 'iid' if 'IID' in partition_type else 'noniid'
        st.session_state.clients_data = partition_data(
            st.session_state.train_x,
            st.session_state.train_y,
            num_clients=num_clients,
            partition_type=p_type,
            alpha=alpha
        )

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Topology & Data",
    "🚀 Simulator Console",
    "🔮 Weight Drift & Privacy",
    "🔬 FL Math Explorer"
])

# ----------------- TAB 1: TOPOLOGY & DATA -----------------
with tab1:
    st.markdown("### Federated Network Topology & Local Clients")
    
    col_topo, col_dist = st.columns([1, 2])
    
    with col_topo:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Network Structure")
        st.write("Clients run local optimizations and upload parameter updates (deltas) instead of raw data.")
        
        # Simple text representation of Client-Server architecture
        topo_html = f"""
        <div style='text-align: center; font-family: sans-serif; color: white; background: #0f172a; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
            <div style='background: #3b82f6; display: inline-block; padding: 0.5rem 1.2rem; border-radius: 8px; font-weight: bold; margin-bottom: 20px;'>
                🌐 Parameter Server
            </div>
            <br>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
        """
        for i in range(num_clients):
            num_samples = len(st.session_state.clients_data[i]['targets'])
            topo_html += f"""
                <div style='background: rgba(52, 211, 153, 0.15); border: 1px solid #34d399; padding: 0.5rem; border-radius: 6px; margin: 5px; font-size: 0.85rem;'>
                    📱 Client {i}<br>
                    <span style='color: #a7f3d0; font-size: 0.75rem;'>{num_samples} samples</span>
                </div>
            """
        topo_html += "</div></div>"
        st.markdown(topo_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_dist:
        # Plot labels distribution bar chart
        fig_dist = plot_client_distributions(st.session_state.clients_data)
        st.plotly_chart(fig_dist, use_container_width=True)

# ----------------- TAB 2: SIMULATOR CONSOLE -----------------
with tab2:
    st.markdown("### Live Simulation Control Center")
    
    col_run_ctrl, col_run_curve = st.columns([1, 2])
    
    with col_run_ctrl:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Experiment Run Settings")
        
        rounds = st.slider("Communication Rounds", min_value=3, max_value=30, value=10, step=1)
        client_participation_ratio = st.slider("Client Participation (per round)", min_value=0.2, max_value=1.0, value=1.0, step=0.1)
        
        st.write("Click below to start the federated training simulation.")
        start_sim = st.button("🚀 Start FL Simulation")
        
        # Save run form
        st.markdown("---")
        st.markdown("##### Save Current Configuration")
        run_name_input = st.text_input("Run Name", value=f"{algorithm} - {'IID' if 'IID' in partition_type else 'Non-IID'}")
        save_run_btn = st.button("💾 Save Run Metrics")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        sim_status = st.empty()
        sim_prog = st.empty()
        
    with col_run_curve:
        curve_holder = st.empty()
        
    if start_sim:
        sim_status.info("Initializing parameter server and clients...")
        
        # 1. Instantiate Server
        test_x = st.session_state.test_x
        test_y = st.session_state.test_y
        
        # Reshape to flatten input if needed by MLP
        input_dim = int(np.prod(test_x.shape[1:]))
        
        server = FLServer(
            test_x, test_y,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=10
        )
        
        # 2. Instantiate Clients
        clients = []
        for i in range(num_clients):
            c_data = st.session_state.clients_data[i]
            clients.append(FLClient(i, c_data['data'], c_data['targets'], batch_size=local_batch_size))
            
        history = {
            'test_loss': [],
            'test_acc': []
        }
        
        drift_snapshots = []
        
        # Log round 0 initial state
        l_loss, l_acc = server.evaluate()
        history['test_loss'].append(l_loss)
        history['test_acc'].append(l_acc)
        
        start_time = time.time()
        
        # 3. RUN COMMUNICATION ROUNDS
        for r in range(1, rounds + 1):
            sim_status.text(f"Round {r}/{rounds} | Local updates in progress...")
            sim_prog.progress(r / rounds)
            
            # Broadcast global weights to all clients
            global_weights = server.get_global_weights()
            
            # Subsample participating clients
            num_participating = max(1, int(num_clients * client_participation_ratio))
            participating_indices = np.random.choice(range(num_clients), size=num_participating, replace=False)
            
            client_locals = []
            client_deltas = []
            client_sizes = []
            
            # Train each client locally
            for idx in participating_indices:
                client = clients[idx]
                c_weights, c_delta, c_loss, c_acc = client.train(
                    server.global_model,
                    lr=local_lr,
                    local_epochs=local_epochs,
                    mu=mu
                )
                client_locals.append(c_weights)
                client_deltas.append(c_delta)
                client_sizes.append(len(client.y_targets))
                
            # Aggregate client deltas on the server
            # Store copy of weights before aggregation to visualize drift
            weights_before_agg = copy.deepcopy(global_weights)
            
            server.aggregate(
                client_deltas, client_sizes,
                algorithm=algorithm,
                momentum_beta=momentum_beta,
                global_lr=1.0,
                dp_noise=dp_noise
            )
            
            # Store weight snapshot (only for first participating clients up to 5 to avoid clutter)
            weights_after_agg = server.get_global_weights()
            drift_snapshots.append({
                'round': r,
                'global_start': weights_before_agg,
                'client_locals': client_locals[:5],
                'global_end': weights_after_agg
            })
            
            # Evaluate global model
            test_loss, test_acc = server.evaluate()
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Update Live Curves
            curve_holder.plotly_chart(plot_fl_metrics(history), use_container_width=True)
            time.sleep(0.01)
            
        elapsed = time.time() - start_time
        sim_status.success(f"Simulation completed! Global model trained in {elapsed:.2f} seconds.")
        
        # Save results in session state
        st.session_state.server = server
        st.session_state.history = history
        st.session_state.drift_snapshots = drift_snapshots
        
    # Display saved metrics
    if st.session_state.history is not None:
        hist = st.session_state.history
        
        st.markdown("---")
        st.markdown("#### Final Global Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{hist["test_loss"][-1]:.4f}</div><div class="metric-lbl">Global Test Loss</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{hist["test_acc"][-1]:.1%}</div><div class="metric-lbl">Global Test Accuracy</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(hist["test_acc"]) - 1}</div><div class="metric-lbl">Total Rounds</div></div>', unsafe_allow_html=True)

    # Save metrics action
    if save_run_btn and st.session_state.history is not None:
        st.session_state.runs_comparison[run_name_input] = st.session_state.history
        st.toast(f"Saved run '{run_name_input}' to memory!", icon="💾")

    # Render comparisons if multiple runs are saved
    if len(st.session_state.runs_comparison) > 1:
        st.markdown("---")
        st.markdown("### Compare Saved Configurations")
        st.write("Compare convergence rates between different runs stored in session memory (e.g. FedAvg vs FedProx).")
        
        col_comp_acc, col_comp_loss = st.columns(2)
        fig_comp_acc, fig_comp_loss = plot_comparison_metrics(st.session_state.runs_comparison)
        
        with col_comp_acc:
            st.plotly_chart(fig_comp_acc, use_container_width=True)
        with col_comp_loss:
            st.plotly_chart(fig_comp_loss, use_container_width=True)

# ----------------- TAB 3: WEIGHT DRIFT & PRIVACY -----------------
with tab3:
    st.markdown("### Latent Weight-Space Analysis & Privacy Metrics")
    
    if st.session_state.history is None or len(st.session_state.drift_snapshots) == 0:
        st.info("⚠️ Please train the model in **Simulator Console** first to view parameter drift charts.")
    else:
        col_drift_ctrl, col_drift_plot = st.columns([1, 3])
        
        with col_drift_ctrl:
            st.markdown("#### Weight Space Analysis")
            st.write("Local client training causes models to drift in different directions due to data heterogeneity. Server aggregation averages them back.")
            
            # Select snapshot round
            drift_rounds = [snap['round'] for snap in st.session_state.drift_snapshots]
            selected_round = st.select_slider("Select Communication Round", options=drift_rounds)
            
            # Find selected snapshot
            snap = next(item for item in st.session_state.drift_snapshots if item['round'] == selected_round)
            
            st.markdown("---")
            st.markdown("#### Privacy Budget (DP Accountant)")
            st.write("Differential Privacy bounds privacy leakage by adding noise to updates.")
            
            # Basic DP Budget Calculation
            # epsilon approx rounds * sqrt(q) * dp_noise... 
            # For simplicity, we show a basic composition bound: Epsilon = Rounds * log(1 + exp(1/noise_multiplier))
            if dp_noise > 0:
                epsilon = selected_round * (1.0 / dp_noise)
                st.markdown(f"- **Noise Scale (sigma):** {dp_noise:.2f}")
                st.markdown(rf"- **Current Privacy Budget ($\epsilon$):** {epsilon:.2f} (Composition Bound)")
                st.markdown("- **Privacy Level:** Strong Protection" if epsilon < 5 else "- **Privacy Level:** Moderate Protection")
            else:
                st.markdown("- **Noise Scale (sigma):** 0.00")
                st.markdown(r"- **Current Privacy Budget ($\epsilon$):** $\infty$ (No privacy guarantees)")
                
        with col_drift_plot:
            fig_drift = plot_weight_drift(
                snap['global_start'],
                snap['client_locals'],
                snap['global_end']
            )
            st.plotly_chart(fig_drift, use_container_width=True)

# ----------------- TAB 4: FL MATH EXPLORER -----------------
with tab4:
    st.markdown("### Core Algorithms & Mathematical Formulations")
    st.write("Learn the mathematics underlying modern Federated Learning algorithms implemented in this sandbox.")
    
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Federated Averaging (FedAvg)")
    st.write("Proposed by McMahan et al., FedAvg aggregates local updates by performing a weighted average based on client sample sizes.")
    st.latex(r"w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k")
    st.write(r"Where $n_k$ is the number of samples on client $k$, $N = \sum n_k$, and $w_{t+1}^k$ is the local model weight vector after training.")
    
    with st.expander("Show FedAvg Aggregation Code"):
        st.code("""
# Central Server aggregation step (FedAvg)
total_samples = sum(client_sizes)
aggregated_delta = {k: np.zeros_like(v) for k, v in global_weights.items()}

# Weighted average sum
for delta, size in zip(client_deltas, client_sizes):
    weight = size / total_samples
    for k in aggregated_delta.keys():
        aggregated_delta[k] += delta[k] * weight

# Update global weights
new_weights = {k: global_weights[k] + aggregated_delta[k] for k in global_weights.keys()}
        """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Federated Proximal Aggregation (FedProx)")
    st.write("Proposed by Li et al., FedProx adds a proximal regularization term to the local client objective. This stabilizes learning on highly non-IID/skewed data.")
    st.latex(r"\mathcal{L}_{\text{prox}}(w) = \mathcal{L}(w) + \frac{\mu}{2} \| w - w_t \|^2")
    st.write(r"Where $w$ is the local client weights, $w_t$ is the current global model, and $\mu$ controls the penalty strength.")
    
    with st.expander("Show FedProx Local Regularization Code"):
        st.code("""
# Local Client training loop loss modification
optimizer.zero_grad()
outputs = local_model(batch_x)
loss = criterion(outputs, batch_y)

if mu > 0.0:
    prox_penalty = 0.0
    for name, param in local_model.named_parameters():
        g_weight = global_weights[name] # central global weights
        prox_penalty += torch.sum((param - g_weight) ** 2)
        
    loss = loss + (mu / 2.0) * prox_penalty

loss.backward()
optimizer.step()
        """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🔏 Local Differential Privacy (DP)")
    st.write("differential privacy injects calibrated noise into local parameter updates before aggregation to prevent the server from reconstructing client training samples.")
    st.latex(r"\Delta_{\text{noisy}} = \Delta + \mathcal{N}\left(0, \sigma^2 S^2 \mathbf{I}\right)")
    st.write(r"Where $\Delta = w_{\text{local}} - w_{\text{global}}$, $\sigma$ is the noise multiplier, and $S$ is the update sensitivity.")
    
    with st.expander("Show Differential Privacy Noise Code"):
        st.code("""
# Adding Gaussian Noise to client updates
def add_dp_noise(delta_weights, noise_multiplier, sensitivity=0.1):
    if noise_multiplier <= 0:
        return delta_weights
        
    noisy_delta = {}
    for k, v in delta_weights.items():
        std = noise_multiplier * sensitivity
        noise = np.random.normal(loc=0.0, scale=std, size=v.shape)
        noisy_delta[k] = v + noise
        
    return noisy_delta
        """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
