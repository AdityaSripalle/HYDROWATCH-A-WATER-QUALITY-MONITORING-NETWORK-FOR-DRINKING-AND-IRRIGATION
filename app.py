import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier, CatBoostRegressor
from io import BytesIO
from PIL import Image

# Global variables
best_model = None
scaler = None
label_encoder = None
global_target_type = "classification"  # Default to classification

# Load Dataset
def load_data(target_type="classification"):
    try:
        df = pd.read_csv(r"C:\Users\adity\OneDrive\Documents\mini\DataSet.csv")
    except FileNotFoundError:
        return None, None, None, None, " Error: 'Dataset.csv' not found!"

    features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
    target = 'Water Quality Classification' if target_type == "classification" else 'WQI'

    if target not in df.columns:
        return None, None, None, None, " Error: Target column missing!"

    df_cleaned = df[features + [target]].copy()
    df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)

    global label_encoder
    if target_type == "classification":
        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

    return df_cleaned, features, target, label_encoder, " Data Loaded Successfully!"

# Train Models
def train_models(target_type="classification"):
    global global_target_type
    global_target_type = target_type

    df_cleaned, features, target, label_encoder, status = load_data(target_type)
    if df_cleaned is None:
        return None, None, status

    global best_model, scaler

    X = df_cleaned[features]
    y = df_cleaned[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if target_type == "classification":
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(kernel='linear', random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, num_class=3, random_state=42),
            'QDA': QuadraticDiscriminantAnalysis(),
            'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.3, depth=5, random_state=42, verbose=0),
        }
    else:
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Support Vector Machine': SVR(kernel='linear'),
            'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
            'Linear Regression': LinearRegression(),
            'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0),
        }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if target_type == "classification":
            results.append({
                'Model Name': name,
                'Training Accuracy': round(accuracy_score(y_train, y_train_pred), 4),
                'Testing Accuracy': round(accuracy_score(y_test, y_test_pred), 4),
                'Precision': round(precision_score(y_test, y_test_pred, average='weighted', zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_test_pred, average='weighted'), 4),
                'R2 Score': round(r2_score(y_test, y_test_pred), 4),
                'MAE': round(mean_absolute_error(y_test, y_test_pred), 4)
            })
        else:
            results.append({
                'Model Name': name,
                'Training MAE': round(mean_absolute_error(y_train, y_train_pred), 4),
                'Testing MAE': round(mean_absolute_error(y_test, y_test_pred), 4),
                'R2 Score': round(r2_score(y_test, y_test_pred), 4)
            })

    results_df = pd.DataFrame(results)
    best_model_name = (
        results_df.loc[results_df['Training Accuracy'].idxmax(), 'Model Name']
        if target_type == "classification"
        else results_df.loc[results_df['Testing MAE'].idxmin(), 'Model Name']
    )
    best_model = models[best_model_name]

    return results_df, best_model_name, "🏆 Best Model Selected: " + best_model_name

# Prediction Function
def predict_water_quality(*features):
    input_data = np.array(features).reshape(1, -1)

    if best_model is not None and scaler is not None:
        input_data_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_data_scaled)

        if global_target_type == "classification":
            predicted_class = label_encoder.inverse_transform(prediction)
            return f"Predicted Water Quality Class: {predicted_class[0]}", f"Water Quality Class: {predicted_class[0]}"
        else:
            return f"Predicted Water Quality Index (WQI): {prediction[0]:.2f}", f"WQI: {prediction[0]:.2f}"
    else:
        return "Error: Model is not trained yet. Please train the model first.", "Error: Model is not trained yet."

# Visualization - Comparison of Metrics
def visualize_model_comparison(results_df):
    import matplotlib
    matplotlib.use('Agg')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['Precision', 'F1 Score', 'R2 Score', 'MAE']
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        sns.barplot(x='Model Name', y=metric, data=results_df, ax=ax, palette='coolwarm')
        ax.set_title(f"{metric} Comparison Across Models")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 8), textcoords='offset points')

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# Accuracy Plot
def visualize_accuracy_comparison(results_df):
    import matplotlib
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.set_index('Model Name')[['Training Accuracy', 'Testing Accuracy']].plot(kind='bar', ax=ax, colormap='coolwarm')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Testing Accuracy for Different Models")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return Image.open(img)

# Gradio UI
with gr.Blocks() as demo:
    with gr.Tabs():
        # Tab 1: Train and Predict
        with gr.TabItem("Train, Table, Prediction"):
            gr.Markdown("# 💧 Water Quality Prediction")

            with gr.Row():
                target_type_input = gr.Dropdown(["classification", "regression"], label="Select Target Type", value="classification")
                train_button = gr.Button("🔄 Train Model")

            with gr.Row():
                results_df = gr.Dataframe(label="📊 Model Performance Table")
                output_text = gr.Textbox(label="Status")

            train_button.click(
                fn=lambda target_type: train_models(target_type),
                inputs=[target_type_input],
                outputs=[results_df, output_text]
            )

            gr.Markdown("### 🔮 Predict Water Quality")
            input_fields = [gr.Number(label=feature, value=0.0) for feature in ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']]

            with gr.Row():
                predict_button = gr.Button("🔮 Predict")
                predict_output = gr.Textbox(label="Prediction Result", interactive=False)
                predict_class_output = gr.Textbox(label="Water Quality Classification", interactive=False)

            predict_button.click(
                fn=predict_water_quality,
                inputs=input_fields,
                outputs=[predict_output, predict_class_output]
            )

        # Tab 2: Model Visualization
        with gr.TabItem("Model Visualization"):
            gr.Markdown("### 📊 Model Comparison")

            def load_visualization():
                df, _, _ = train_models("classification")
                return visualize_model_comparison(df)

            vis_button = gr.Button("📊 Generate Model Comparison")
            vis_output = gr.Image(label="Model Comparison")
            vis_button.click(fn=load_visualization, inputs=[], outputs=vis_output)

        # Tab 3: Accuracy Visualization
        with gr.TabItem("Results"):
            gr.Markdown("### 📈 Training vs Testing Accuracy")

            def load_accuracy():
                df, _, _ = train_models("classification")
                return visualize_accuracy_comparison(df)

            acc_button = gr.Button("📈 Generate Accuracy Plot")
            acc_output = gr.Image(label="Accuracy Comparison")
            acc_button.click(fn=load_accuracy, inputs=[], outputs=acc_output)

# Run app
if __name__ == "__main__":
    demo.launch()
