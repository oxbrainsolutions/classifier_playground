import streamlit as st
from streamlit_echarts import st_echarts
import numpy as np
import time
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


text_media_query_functions1 = '''
<style>
@media (max-width: 1024px) {
  p.text {
      font-size: 4em;
  }
}
</style>
'''

def change_callback3():
    st.session_state.submit_confirm2 = False


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
            colorscale=["#5007E3", "#03A9F4"],
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
            colorscale=["#5007E3", "#03A9F4"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(train_data).add_trace(test_data).update_xaxes(range=[x_min, x_max]).update_yaxes(range=[y_min, y_max]).update_layout(xaxis_title=dict(text="X1", font=dict(size=16, family="sans-serif", color="#FAFAFA")), xaxis=dict(tickfont=dict(size=14, family="sans-serif", color="#FAFAFA")), yaxis_title=dict(text="X2", font=dict(size=16, family="sans-serif", color="#FAFAFA")), yaxis=dict(tickfont=dict(size=14, family="sans-serif", color="#FAFAFA")))

    fig.update_xaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_yaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_layout(autosize=True, height=500, width = 500, margin=dict(l=5, r=10, b=0, t=10), legend=dict(font=dict(size=14, family="sans-serif", color="#FAFAFA"), bgcolor="rgba(0, 0, 0, 0)", orientation="h", yanchor="top", y=1, xanchor="right", x=1))
    return fig

def plot_scatter_decision_boundary(model, x_train, y_train, x_test, y_test):
    d = x_train.shape[1]
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p) for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)
  
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
            colorscale=["#5007E3", "#03A9F4"],
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
            colorscale=["#5007E3", "#03A9F4"],
            line=dict(color="black", width=2),
        ),
    )

    heatmap = go.Heatmap(
            x=xx[0],
            y=y_,
            z=Z,
            colorscale=["#5007E3", "#03A9F4"],
            opacity=0.4,
            showscale=False,
        )
  
    fig.add_trace(heatmap).add_trace(train_data).add_trace(test_data).update_xaxes(range=[x_min, x_max]).update_yaxes(range=[y_min, y_max]).update_layout(xaxis_title=dict(text="X1", font=dict(size=16, family="sans-serif", color="#FAFAFA")), xaxis=dict(tickfont=dict(size=14, family="sans-serif", color="#FAFAFA")), yaxis_title=dict(text="X2", font=dict(size=16, family="sans-serif", color="#FAFAFA")), yaxis=dict(tickfont=dict(size=14, family="sans-serif", color="#FAFAFA")))

    fig.update_xaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_yaxes(showline=True, showgrid=False, zeroline=False, linecolor = '#FAFAFA', linewidth = 2.5, mirror = True)
    fig.update_layout(autosize=True, height=500, width = 500, margin=dict(l=5, r=10, b=0, t=10), legend=dict(font=dict(size=14, family="sans-serif", color="#FAFAFA"), bgcolor="rgba(0, 0, 0, 0)", orientation="h", yanchor="top", y=1, xanchor="right", x=1))
    return fig

color1 = "#5007E3"
color2 = "#03A9F4"
col_cmap = clr.LinearSegmentedColormap.from_list(name="", colors=[color1, color2])
num_steps = 1000
values = np.linspace(0, 1, num_steps)
step_values = np.linspace(0, 1, num_steps)
colors = col_cmap(values)
step_colors = [clr.rgb2hex(color) for color in colors]


def create_gauge(num_value, label, key):
        option = {
        "series": [
            {
            "type": 'gauge',
            "min": 0,
            "max": 1,
            "center": ['50%', '50%'],
            "radius": '100%',
            "splitNumber": 5,
            "progress": {
                "show": False,
              },
             "anchor": {},
            "axisLine": {
              "lineStyle": {
                "width": 20,
                "color": [[step_values[i], step_colors[i]] for i in range(num_steps - 1)]
                }
              },
            "axisTick": {
                "splitNumber": 1,
                "length": 25,
                "distance": -20,
                "lineStyle": {
                "color": 'auto', "width": 3
                }
            },
            "splitLine": {
                "show": False
            },
            "axisLabel": {
                "distance": 10,
                "textStyle": {
                    "fontFamily": 'sans-serif',
                    "color": '#FAFAFA', "fontSize": 12
                },
            },
            "title": {
            "offsetCenter": [0, '-15%'],
            "fontSize": 20,
            "color": '#FCBC24',
            "fontFamily": 'sans-serif',
              },
            "detail": {
              "valueAnimation": True,
              "formatter": '{}'.format(num_value),
              "color": 'auto',
              "offsetCenter": [0, '30%'],
              "fontSize": 35,
              "fontFamily": 'sans-serif',
              },
            "pointer": {
                "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                "width": 20,
                "itemStyle": {"color": "#FCBC24"},
                "offsetCenter": [0, '-84%'],
                "length": '15%',
            },
            "data": [
              {
                "value": num_value,
                "name": '{}'.format(label),
                },
              ]
            }
          ]
        }
        st_echarts(option, height="230px", key=key)

def convert_rating(value):
  if value < 0.33:
    rating = "Weak"
  elif value < 0.66:
    rating = "Moderate"
  else:
    rating = "Strong"
  return rating

def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test

def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 4)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 4)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 4)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 4)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration


def lr_param_selector():
    solver = "saga"
    max_iter = 1000
    
    penalty_options = ["", "None", "Lasso", "Ridge", "Elastic Net"]
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Regularization Method</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_penalty = st.selectbox(label="", label_visibility="collapsed", options=penalty_options, format_func=lambda x: "Select Method" if x == "" else x, key="key_lr1", on_change=change_callback3)
    penalty_options_update = ["", "none", "l1", "l2", "elasticnet"][penalty_options.index(user_penalty)]

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Complexity Constraint</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_constraint = st.number_input(label="", label_visibility="collapsed", min_value=0.1, max_value=2.0, step=0.1, value=1.0, key="key_lr2", on_change=change_callback3)
    C = np.round(user_constraint, 3)

    if user_penalty == "Elastic Net":
      l1_ratio = 0.5
      params = {"solver": solver, "penalty": penalty_options_update, "C": C, "max_iter": max_iter, "l1_ratio": l1_ratio}
    else:
      params = {"solver": solver, "penalty": penalty_options_update, "C": C, "max_iter": max_iter}
    
    model = LogisticRegression(**params)
    return model

def nb_param_selector():
    model = GaussianNB()
    return model

def lda_param_selector():
    model = LinearDiscriminantAnalysis()
    return model

def qda_param_selector():
    model = QuadraticDiscriminantAnalysis()
    return model

def knn_param_selector():
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Neighbors</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_neighbors = st.number_input(label="", label_visibility="collapsed", min_value=3, max_value=20, step=1, value=5, key="key_knn1", on_change=change_callback3)
    
    metric = "euclidean"
    params = {"n_neighbors": user_neighbors, "metric": metric}

    model = KNeighborsClassifier(**params)
    return model
  
def nn_param_selector():
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Hidden Layers</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_hidden_layers = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=5, step=1, value=1, key="key_nn1", on_change=change_callback3)

    hidden_layer_sizes = []

    for i in range(user_hidden_layers):
        text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Neurons in Layer {}</span></p>'.format(i+1)
        st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
        user_n_neurons = st.number_input(label="", label_visibility="collapsed", min_value=5, max_value=200, step=5, value=5, key="key_nn{}".format(i+2), on_change=change_callback3)
        hidden_layer_sizes.append(user_n_neurons)

    hidden_layer_sizes = tuple(hidden_layer_sizes)
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    model = MLPClassifier(**params)
    return model

def svm_param_selector():
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Cost Value</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_C = st.number_input(label="", label_visibility="collapsed", min_value=0.01, max_value=2.00, step=1.0, value=0.01, key="key_svm1", on_change=change_callback3)

    kernel_options = ["", "Linear", "Polynomial", "Radial", "Sigmoid"]
    user_kernel = st.selectbox(label="", label_visibility="collapsed", options=kernel_options, format_func=lambda x: "Select Kernel" if x == "" else x, key="key_svm2", on_change=change_callback3)
    kernel_options_update = ["", "linear", "poly", "rbf", "sigmoid"][kernel_options.index(user_kernel)]

    params = {"C": user_C, "kernel": kernel_options_update}
    model = SVC(**params)
    return model

def ct_param_selector():

    criterion = "gini"

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Maximum Depth</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_max_depth = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=50, step=1, value=5, key="key_ct1", on_change=change_callback3)

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Minimum Sample Split</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_min_samples_split = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=20, step=1, value=2, key="key_ct2", on_change=change_callback3)

    max_features = "sqrt"

    params = {
        "criterion": criterion,
        "max_depth": user_max_depth,
        "min_samples_split": user_min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeClassifier(**params)
    return model


def rf_param_selector():

    criterion = "gini"

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Estimators</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_n_estimators = st.number_input(label="", label_visibility="collapsed", min_value=50, max_value=300, step=10, value=100, key="key_rf1", on_change=change_callback3)

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Maximum Depth</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_max_depth = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=50, step=1, value=5, key="key_rf2", on_change=change_callback3)

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Minimum Sample Split</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_min_samples_split = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=20, step=1, value=2, key="key_rf3", on_change=change_callback3)

    max_features = "sqrt"
  
    params = {
        "criterion": criterion,
        "n_estimators": user_n_estimators,
        "max_depth": user_max_depth,
        "min_samples_split": user_min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    return model

def ad_param_selector():

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Learning Rate</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_learning_rate = st.slider(label="", label_visibility="collapsed", min_value=0.001, max_value=0.5, step=0.005, value=0.1, key="key_ab1", on_change=change_callback3)
  
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Estimators</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_n_estimators = st.number_input(label="", label_visibility="collapsed", min_value=50, max_value=300, step=10, value=100, key="key_ab2", on_change=change_callback3)

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Maximum Depth</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_max_depth = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=50, step=1, value=5, key="key_ab3", on_change=change_callback3)

    params = {
        "learning_rate": user_learning_rate,
        "n_estimators": user_n_estimators,
        "max_depth": user_max_depth,
    }

    model = AdaBoostClassifier(**params)
  
  return model

def gb_param_selector():

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Learning Rate</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_learning_rate = st.slider(label="", label_visibility="collapsed", min_value=0.001, max_value=0.5, step=0.005, value=0.1, key="key_gb1", on_change=change_callback3)
  
    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Number of Estimators</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_n_estimators = st.number_input(label="", label_visibility="collapsed", min_value=50, max_value=300, step=10, value=100, key="key_gb2", on_change=change_callback3)

    text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Maximum Depth</span></p>'
    st.markdown(text_media_query_functions1 + text, unsafe_allow_html=True)
    user_max_depth = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=50, step=1, value=5, key="key_gb3", on_change=change_callback3)

    params = {
        "learning_rate": user_learning_rate,
        "n_estimators": user_n_estimators,
        "max_depth": user_max_depth,
    }

    model = GradientBoostingClassifier(**params)
    return model
