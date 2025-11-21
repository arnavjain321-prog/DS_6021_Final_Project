import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("final_master_dataset.csv")
df = df.dropna(subset=["trifecta_2024"])

# Feature definitions
num_features = [
    "poverty_rate",
    "median_household_income",
    "white_pct", "black_pct", "native_pct", "asian_pct", "pacific_pct", "two_plus_pct",
    "grocery_cost_index",
    "minimum_wage",
    "RUCC",
    "population",
    "participants_per_1000",
]

cat_features = [
    "trifecta_2024",
    "usda_snap_region",
    "min_wage_tier",
    "rural_urban_category",
]

target_reg = "benefits_per_person"
target_clf = "snap_policy_class"

X = df[num_features + cat_features]
y_reg = df[target_reg]
y_clf = df[target_clf]

# Precompute numeric means for use in custom prediction form
feature_means = df[num_features].mean()

# =====================================
# PREPROCESSOR (numeric + categorical)
# =====================================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# =====================================
# TRAIN / TEST SPLITS
# =====================================
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# =====================================
# REGRESSION MODELS: Linear, Ridge, Lasso
# =====================================
linreg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
linreg_model.fit(X_train_reg, y_train_reg)

y_pred_train_lin = linreg_model.predict(X_train_reg)
y_pred_test_lin = linreg_model.predict(X_test_reg)

lin_train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_pred_train_lin))
lin_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_test_lin))
lin_train_r2 = r2_score(y_train_reg, y_pred_train_lin)
lin_test_r2 = r2_score(y_test_reg, y_pred_test_lin)

ridge_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=10.0))
])
ridge_model.fit(X_train_reg, y_train_reg)
y_pred_test_ridge = ridge_model.predict(X_test_reg)
ridge_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_test_ridge))
ridge_test_r2 = r2_score(y_test_reg, y_pred_test_ridge)

lasso_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", Lasso(alpha=1.0, max_iter=5000))
])
lasso_model.fit(X_train_reg, y_train_reg)
y_pred_test_lasso = lasso_model.predict(X_test_reg)
lasso_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_test_lasso))
lasso_test_r2 = r2_score(y_test_reg, y_pred_test_lasso)

# Scatter fig: Actual vs Predicted (Linear)
fig_reg_scatter = px.scatter(
    x=y_test_reg,
    y=y_pred_test_lin,
    labels={"x": "Actual benefits_per_person", "y": "Predicted benefits_per_person"},
    title="Actual vs Predicted (Linear Regression)"
)
fig_reg_scatter.add_trace(
    go.Scatter(
        x=[y_test_reg.min(), y_test_reg.max()],
        y=[y_test_reg.min(), y_test_reg.max()],
        mode="lines",
        name="Ideal",
        line=dict(dash="dash")
    )
)
fig_reg_scatter.update_layout(
    height=400,
    margin=dict(t=40, b=40, l=40, r=20)
)

# =====================================
# CLASSIFICATION MODELS: Logistic, KNN, MLP
# =====================================
log_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=800, multi_class="multinomial"))
])
log_model.fit(X_train_clf, y_train_clf)

y_pred_train_log = log_model.predict(X_train_clf)
y_pred_test_log = log_model.predict(X_test_clf)
log_train_acc = accuracy_score(y_train_clf, y_pred_train_log)
log_test_acc = accuracy_score(y_test_clf, y_pred_test_log)

# Confusion matrix for logistic
cm_log = confusion_matrix(y_test_clf, y_pred_test_log, labels=sorted(y_clf.unique()))
cm_log_fig = px.imshow(
    cm_log,
    x=sorted(y_clf.unique()),
    y=sorted(y_clf.unique()),
    text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    title="Confusion Matrix – Logistic Regression"
)
cm_log_fig.update_layout(
    height=400,
    margin=dict(t=40, b=40, l=40, r=20)
)

# KNN
knn_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", KNeighborsClassifier(n_neighbors=5))
])
knn_model.fit(X_train_clf, y_train_clf)
y_pred_test_knn = knn_model.predict(X_test_clf)
knn_test_acc = accuracy_score(y_test_clf, y_pred_test_knn)

# MLP
mlp_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", MLPClassifier(hidden_layer_sizes=(32, 16),
                            activation="relu",
                            max_iter=2000,
                            random_state=42))
])
mlp_model.fit(X_train_clf, y_train_clf)
y_pred_test_mlp = mlp_model.predict(X_test_clf)
mlp_test_acc = accuracy_score(y_test_clf, y_pred_test_mlp)

# =====================================
# CLUSTERING + PCA
# =====================================
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[num_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)
df["cluster_k3"] = cluster_labels

fig_kmeans = px.scatter(
    df,
    x="poverty_rate",
    y="participants_per_1000",
    color="cluster_k3",
    hover_name="state",
    title="K-means Clusters (k=3) – Poverty vs Participants per 1000"
)
fig_kmeans.update_layout(
    height=400,
    margin=dict(t=40, b=40, l=40, r=20)
)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "snap_policy_class": df["snap_policy_class"],
    "cluster_k3": df["cluster_k3"],
    "state": df["state"]
})

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="snap_policy_class",
    symbol="cluster_k3",
    hover_name="state",
    title=f"PCA (2D) – Explained variance: {pca.explained_variance_ratio_.sum():.2f}"
)
fig_pca.update_layout(
    height=400,
    margin=dict(t=40, b=40, l=40, r=20)
)

# =====================================
# DASH APP
# =====================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Container([
    html.Br(),
    html.H2("SNAP ML Dashboard"),
    html.Hr(),

    dbc.Tabs([

        # --------- TAB 1: EDA ---------
        dbc.Tab(label="EDA", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Histogram variable:"),
                    dcc.Dropdown(
                        id="eda-hist-var",
                        options=[{"label": c, "value": c} for c in [
                            "poverty_rate",
                            "benefits_per_person",
                            "grocery_cost_index",
                            "participants_per_1000",
                            "minimum_wage",
                            "median_household_income"
                        ]],
                        value="poverty_rate"
                    ),
                    dcc.Graph(id="eda-hist-fig", style={"height": "450px"})
                ], width=6),

                dbc.Col([
                    html.Label("Scatter: X vs Y (colored by policy class)"),
                    dcc.Dropdown(
                        id="eda-x-var",
                        options=[{"label": c, "value": c} for c in num_features],
                        value="poverty_rate"
                    ),
                    dcc.Dropdown(
                        id="eda-y-var",
                        options=[{"label": c, "value": c} for c in [
                            "participants_per_1000",
                            "benefits_per_person",
                            "median_household_income"
                        ]],
                        value="participants_per_1000"
                    ),
                    dcc.Graph(id="eda-scatter-fig", style={"height": "450px"})
                ], width=6),
            ]),

            html.Br(),
            html.H4("Benefits per Person by SNAP Policy Class"),
            dcc.Graph(
                figure=px.box(
                    df,
                    x="snap_policy_class",
                    y="benefits_per_person",
                    title="Benefits per Person by SNAP Policy Class"
                ).update_layout(
                    height=400,
                    margin=dict(t=40, b=40, l=40, r=20)
                ),
                style={"height": "450px"}
            )
        ]),

        # --------- TAB 2: Regression ---------
        dbc.Tab(label="Regression", children=[
            html.Br(),
            html.H4("Linear Regression – Predicting Benefits per Person"),
            dbc.Row([
                dbc.Col([
                    html.Ul([
                        html.Li(f"Train RMSE: {lin_train_rmse:.2f}"),
                        html.Li(f"Test RMSE: {lin_test_rmse:.2f}"),
                        html.Li(f"Train R²: {lin_train_r2:.3f}"),
                        html.Li(f"Test R²: {lin_test_r2:.3f}")
                    ])
                ], width=6),
                dbc.Col([
                    html.H5("Ridge (alpha=10) – Test:"),
                    html.Ul([
                        html.Li(f"Test RMSE: {ridge_test_rmse:.2f}"),
                        html.Li(f"Test R²: {ridge_test_r2:.3f}")
                    ]),
                    html.H5("Lasso (alpha=1) – Test:"),
                    html.Ul([
                        html.Li(f"Test RMSE: {lasso_test_rmse:.2f}"),
                        html.Li(f"Test R²: {lasso_test_r2:.3f}")
                    ])
                ], width=6),
            ]),
            html.Br(),
            html.H5("Actual vs Predicted (Test Set) – Linear Regression"),
            dcc.Graph(figure=fig_reg_scatter, style={"height": "450px"})
        ]),

        # --------- TAB 3: Classification ---------
        dbc.Tab(label="Classification", children=[
            html.Br(),
            html.H4("Classification Models – Predict SNAP Policy Class"),
            dbc.Row([
                dbc.Col([
                    html.H5("Logistic Regression:"),
                    html.Ul([
                        html.Li(f"Train Accuracy: {log_train_acc:.3f}"),
                        html.Li(f"Test Accuracy: {log_test_acc:.3f}")
                    ]),
                    html.H5("KNN (k=5):"),
                    html.Ul([
                        html.Li(f"Test Accuracy: {knn_test_acc:.3f}")
                    ]),
                    html.H5("MLP Neural Network:"),
                    html.Ul([
                        html.Li(f"Test Accuracy: {mlp_test_acc:.3f}")
                    ])
                ], width=4),
                dbc.Col([
                    html.H5("Logistic Regression – Confusion Matrix (Test Set)"),
                    dcc.Graph(figure=cm_log_fig, style={"height": "450px"})
                ], width=8),
            ])
        ]),

        # --------- TAB 4: Clustering / PCA ---------
        dbc.Tab(label="Clustering & PCA", children=[
            html.Br(),
            html.H4("K-means Clustering (k=3)"),
            dcc.Graph(figure=fig_kmeans, style={"height": "450px"}),
            html.Br(),
            html.H4("PCA (2D Projection)"),
            dcc.Graph(figure=fig_pca, style={"height": "450px"})
        ]),

        # --------- TAB 5: Predictions (Custom Inputs) ---------
        dbc.Tab(label="Predictions", children=[
            html.Br(),
            html.H4("Custom Model Predictions"),

            dbc.Row([
                dbc.Col([
                    html.Label("Poverty Rate"),
                    dcc.Slider(5, 35, 1, value=float(feature_means["poverty_rate"]),
                               id="input-poverty"),
                    html.Br(),

                    html.Label("Median Household Income"),
                    dcc.Input(id="input-income", type="number",
                              value=float(feature_means["median_household_income"])),
                    html.Br(), html.Br(),

                    html.Label("Grocery Cost Index"),
                    dcc.Slider(70, 160, 1,
                               value=float(feature_means["grocery_cost_index"]),
                               id="input-grocery"),
                    html.Br(),

                    html.Label("Minimum Wage"),
                    dcc.Input(id="input-minwage", type="number",
                              value=float(feature_means["minimum_wage"])),
                    html.Br(), html.Br(),

                    html.Label("Participants per 1000"),
                    dcc.Input(id="input-participants", type="number",
                              value=float(feature_means["participants_per_1000"])),
                    html.Br(), html.Br(),

                    html.Label("RUCC (Rural-Urban Continuum Code)"),
                    dcc.Slider(1, 9, 1,
                               value=int(round(feature_means["RUCC"])),
                               id="input-rucc"),
                    html.Br(),

                    html.Label("Population"),
                    dcc.Input(id="input-pop", type="number",
                              value=int(feature_means["population"])),
                ], width=6),

                dbc.Col([
                    html.Label("Trifecta 2024"),
                    dcc.Dropdown(
                        id="input-trifecta",
                        options=[{"label": x, "value": x} for x in sorted(df["trifecta_2024"].unique())],
                        value=sorted(df["trifecta_2024"].unique())[0]
                    ),
                    html.Br(),

                    html.Label("USDA SNAP Region"),
                    dcc.Dropdown(
                        id="input-region",
                        options=[{"label": x, "value": x} for x in sorted(df["usda_snap_region"].unique())],
                        value=sorted(df["usda_snap_region"].unique())[0]
                    ),
                    html.Br(),

                    html.Label("Minimum Wage Tier"),
                    dcc.Dropdown(
                        id="input-tier",
                        options=[{"label": x, "value": x} for x in sorted(df["min_wage_tier"].unique())],
                        value=sorted(df["min_wage_tier"].unique())[0]
                    ),
                    html.Br(),

                    html.Label("Rural/Urban Category"),
                    dcc.Dropdown(
                        id="input-rural",
                        options=[{"label": x, "value": x} for x in sorted(df["rural_urban_category"].unique())],
                        value=sorted(df["rural_urban_category"].unique())[0]
                    ),
                ], width=6),
            ]),

            html.Br(),
            html.H4("Prediction Output"),
            html.Div(id="pred-custom-output", style={"fontSize": 18})
        ]),

    ])
], fluid=True)

# =====================================
# CALLBACKS
# =====================================

# EDA: histogram
@app.callback(
    Output("eda-hist-fig", "figure"),
    Input("eda-hist-var", "value")
)
def update_hist(var):
    fig = px.histogram(
        df,
        x=var,
        nbins=10,
        title=f"Histogram of {var}"
    )
    fig.update_layout(
        height=400,
        margin=dict(t=40, b=40, l=40, r=20)
    )
    return fig

# EDA: scatter
@app.callback(
    Output("eda-scatter-fig", "figure"),
    Input("eda-x-var", "value"),
    Input("eda-y-var", "value")
)
def update_eda_scatter(xvar, yvar):
    fig = px.scatter(
        df,
        x=xvar,
        y=yvar,
        color="snap_policy_class",
        hover_name="state",
        title=f"{yvar} vs {xvar} (colored by SNAP policy class)"
    )
    fig.update_layout(
        height=400,
        margin=dict(t=40, b=40, l=40, r=20)
    )
    return fig

# Predictions tab: custom inputs
@app.callback(
    Output("pred-custom-output", "children"),
    [
        Input("input-poverty", "value"),
        Input("input-income", "value"),
        Input("input-grocery", "value"),
        Input("input-minwage", "value"),
        Input("input-participants", "value"),
        Input("input-rucc", "value"),
        Input("input-pop", "value"),
        Input("input-trifecta", "value"),
        Input("input-region", "value"),
        Input("input-tier", "value"),
        Input("input-rural", "value"),
    ]
)
def predict_custom(pov, inc, gro, wage, part, rucc, pop, trifecta, region, tier, rural):
    # Start from dataset means for all numeric features
    base = feature_means.to_dict()

    # Override with user inputs
    base["poverty_rate"] = pov
    base["median_household_income"] = inc
    base["grocery_cost_index"] = gro
    base["minimum_wage"] = wage
    base["participants_per_1000"] = part
    base["RUCC"] = rucc
    base["population"] = pop

    # Create single-row DataFrame
    row = pd.DataFrame([base])

    # Add categoricals
    row["trifecta_2024"] = trifecta
    row["usda_snap_region"] = region
    row["min_wage_tier"] = tier
    row["rural_urban_category"] = rural

    # Ensure col order matches training
    row = row[num_features + cat_features]

    # Regression prediction
    reg_pred = linreg_model.predict(row)[0]

    # Classification prediction
    class_pred = log_model.predict(row)[0]
    proba = log_model.predict_proba(row)[0]
    classes = log_model.named_steps["model"].classes_
    prob_dict = {c: p for c, p in zip(classes, proba)}

    return html.Div([
        html.P(f"Predicted benefits_per_person: {reg_pred:.2f}"),
        html.P(f"Predicted SNAP policy class: {class_pred}"),
        html.P("Prediction probabilities:"),
        html.Ul([
            html.Li(f"{cls}: {prob_dict[cls]:.3f}") for cls in sorted(prob_dict.keys())
        ])
    ])

# =====================================
# RUN APP
# =====================================
server = app.server  # for deployment (Render / gunicorn)

if __name__ == "__main__":
    app.run(debug=True)
