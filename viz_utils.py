import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Styling Constants
COLOR_PALETTE = px.colors.qualitative.Plotly
COLOR_BG = 'rgba(30, 39, 46, 1.0)'
COLOR_GRID = 'rgba(128, 142, 155, 0.2)'

def plot_client_distributions(client_data):
    """
    Creates a stacked bar chart showing the frequency of label classes per client.
    Helps visualize IID vs. Non-IID Dirichlet data partitioning.
    """
    records = []
    for client_id, c_data in enumerate(client_data):
        targets = c_data['targets']
        classes, counts = np.unique(targets, return_counts=True)
        
        # Populate counts for all 10 classes to ensure consistency
        counts_dict = dict(zip(classes, counts))
        for c in range(10):
            records.append({
                'Client': f"Client {client_id}",
                'Class': f"Digit {c}",
                'Samples': counts_dict.get(c, 0)
            })
            
    df = pd.DataFrame(records)
    
    fig = px.bar(
        df, x='Client', y='Samples', color='Class',
        title="Local Dataset Label Distribution per Client",
        color_discrete_sequence=px.colors.qualitative.Safe,
        barmode='stack'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        legend=dict(title_font=dict(color='#ffffff'), font=dict(color='#ffffff')),
        margin=dict(b=40, l=40, r=40, t=50)
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    fig.update_xaxes(gridcolor=COLOR_GRID)
    
    return fig


def plot_weight_drift(global_start, client_locals, global_end):
    """
    Flattens model parameters and projects them to 2D using PCA.
    Visualizes client models drifting away from the start weights and pulling
    together at the aggregated end weights.
    
    global_start: dict of numpy weights
    client_locals: list of dicts of client numpy weights
    global_end: dict of numpy weights
    """
    def flatten_weights(weights):
        # Flatten all parameter arrays and concatenate into a single vector
        return np.concatenate([v.flatten() for v in weights.values()])
        
    start_vec = flatten_weights(global_start)
    end_vec = flatten_weights(global_end)
    client_vecs = [flatten_weights(c) for c in client_locals]
    
    # Pack into single matrix: Start (1) + Clients (K) + End (1) -> (K + 2) x Params_Dim
    all_vecs = np.vstack([start_vec] + client_vecs + [end_vec])
    
    # Fit PCA to project to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)
    
    start_xy = coords[0]
    client_xys = coords[1:-1]
    end_xy = coords[-1]
    
    # Create Scatter Figure
    fig = go.Figure()
    
    # 1. Add trajectories (Start -> Client Locals) and return vectors
    for idx, c_xy in enumerate(client_xys):
        # Draw line showing drift
        fig.add_trace(go.Scatter(
            x=[start_xy[0], c_xy[0]], y=[start_xy[1], c_xy[1]],
            mode='lines+markers',
            name=f"Client {idx} Drift",
            line=dict(color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], width=1.5, dash='dot'),
            hoverinfo='none',
            showlegend=False
        ))
        # Draw line showing aggregation pull
        fig.add_trace(go.Scatter(
            x=[c_xy[0], end_xy[0]], y=[c_xy[1], end_xy[1]],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.25)', width=1, dash='solid'),
            hoverinfo='none',
            showlegend=False
        ))
        
    # 2. Add starting global weights
    fig.add_trace(go.Scatter(
        x=[start_xy[0]], y=[start_xy[1]],
        mode='markers', name='Starting Global Model',
        marker=dict(size=15, color='#0984e3', symbol='circle', line=dict(width=2, color='#ffffff'))
    ))
    
    # 3. Add client local models endpoints
    for idx, c_xy in enumerate(client_xys):
        fig.add_trace(go.Scatter(
            x=[c_xy[0]], y=[c_xy[1]],
            mode='markers', name=f'Client {idx} Local',
            marker=dict(size=11, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], symbol='diamond', line=dict(width=1, color='#ffffff'))
        ))
        
    # 4. Add aggregated end global weights
    fig.add_trace(go.Scatter(
        x=[end_xy[0]], y=[end_xy[1]],
        mode='markers', name='Aggregated Global Model',
        marker=dict(size=16, color='#2ecc71', symbol='star', line=dict(width=2, color='#ffffff'))
    ))
    
    fig.update_layout(
        title="2D Weight-Space Projection (PCA)",
        xaxis=dict(title="PCA Component 1", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="PCA Component 2", showgrid=True, gridcolor=COLOR_GRID),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        margin=dict(b=40, l=40, r=40, t=50),
        legend=dict(font=dict(color='#ffffff'), borderwidth=0)
    )
    
    return fig


def plot_fl_metrics(history):
    """
    Plots training loss and test accuracy curves across rounds.
    """
    rounds = np.arange(len(history['test_acc']))
    
    fig = go.Figure()
    
    # Loss trace
    fig.add_trace(go.Scatter(
        x=rounds, y=history['test_loss'],
        mode='lines+markers', name='Global Test Loss',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Accuracy trace (Right y-axis)
    fig.add_trace(go.Scatter(
        x=rounds, y=history['test_acc'],
        mode='lines+markers', name='Global Test Accuracy',
        line=dict(color='#2ecc71', width=2),
        yaxis='y2'
    ))
    
    # Layout with dual y-axes
    fig.update_layout(
        title="Federated Convergence History",
        xaxis=dict(title="Communication Rounds", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Test Cross-Entropy Loss", showgrid=True, gridcolor=COLOR_GRID),
        yaxis2=dict(
            title="Test Accuracy",
            overlaying='y',
            side='right',
            range=[0, 1.05],
            showgrid=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(b=40, l=40, r=40, t=65)
    )
    
    return fig


def plot_comparison_metrics(runs):
    """
    Compares multiple simulation runs (e.g. FedAvg vs FedProx) in a single chart.
    runs: dict mapping run_name -> history_dict
    """
    fig_acc = go.Figure()
    fig_loss = go.Figure()
    
    for idx, (name, hist) in enumerate(runs.items()):
        rounds = np.arange(len(hist['test_acc']))
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        
        # Accuracy curves
        fig_acc.add_trace(go.Scatter(
            x=rounds, y=hist['test_acc'],
            mode='lines', name=f"{name} (Acc)",
            line=dict(color=color, width=2)
        ))
        
        # Loss curves
        fig_loss.add_trace(go.Scatter(
            x=rounds, y=hist['test_loss'],
            mode='lines', name=f"{name} (Loss)",
            line=dict(color=color, width=2, dash='dash')
        ))
        
    layout_cfg = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(title="Communication Rounds", showgrid=True, gridcolor=COLOR_GRID),
        margin=dict(b=40, l=40, r=40, t=40)
    )
    
    fig_acc.update_layout(title="Global Accuracy Comparison", yaxis=dict(title="Accuracy", gridcolor=COLOR_GRID), **layout_cfg)
    fig_loss.update_layout(title="Global Loss Comparison", yaxis=dict(title="Loss", gridcolor=COLOR_GRID), **layout_cfg)
    
    return fig_acc, fig_loss
