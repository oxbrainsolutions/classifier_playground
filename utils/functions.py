import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "Spirals":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "Circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "Blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=123,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=123,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test

def plot_scatter(x_train, y_train, x_test, y_test):
    d = x_train.shape[1]
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"colspan": 2}, None]],
        row_heights=[0.7],
    )
    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="Train Data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["#F406E2", "#82F1EB"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="Test Data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["#F406E2", "#82F1EB"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(train_data).add_trace(
        test_data).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2")

    fig.update_xaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_yaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_layout(autosize=True, height=500, width = 500, margin=dict(l=5, r=10, b=0, t=10), legend=dict(orientation="h", yanchor="top", y=1, xanchor="right", x=1))
    return fig

def lr_param_selector():
    solver == "saga"
    
    penalty_options = ["None", "Lasso", "Ridge", "Elatsic Net"]
    user_penalty = st.selectbox(label="", label_visibility="collapsed", options=penalty_options, format_func=lambda x: "Select Regularization" if x == "" else x, key="key7")
    penalty_options_update = ["none", "l1", "l2", "elasticnet"][penalty_options.index(user_penalty)]

    user_constraint = st.number_input(label="", label_visibility="collapsed", min_value=0.1, max_value=2.0, step=0.1, value=1.0, key="key8")
    C = np.round(user_constraint, 3)

    user_constraint = st.number_input(label="", label_visibility="collapsed", min_value=0.1, max_value=2.0, step=0.1, value=1.0, key="key8")
    user_max_iter = st.number_input(label="", label_visibility="collapsed", min_value=100, max_value=2000, step=50, value=100, key="key9")

    params = {"solver": solver, "penalty": penalty_options_update, "C": C, "max_iter": user_max_iter}
    model = LogisticRegression(**params)
    return model

