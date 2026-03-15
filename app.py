# ======================================================
# Flask App: Login + Upload CSV + Cached Feature Extraction + Prediction + MySQL
# ======================================================

from flask import Flask, render_template, request, redirect, url_for, session
import os
import joblib
import numpy as np
import pandas as pd
import pymysql
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ======================================================
# CONFIG
# ======================================================
app = Flask(__name__)
app.secret_key = "securekey123"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)



# ======================================================
# MYSQL CONNECTION
# ======================================================
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "flask_app_db"
}

def get_db_connection():
    return pymysql.connect(**db_config)


# ======================================================
# STATIC LABELS
# ======================================================

# ======================================================
# NLTK SETUP
# ======================================================
download("punkt", quiet=True)
download("stopwords", quiet=True)
download("wordnet", quiet=True)

# ======================================================
# PREPROCESSING FUNCTION
# ======================================================
def upload_dataset(file_path):
    """Load the dataset from a CSV file"""
    df = pd.read_csv(file_path,)
    return df
    
def preprocess_data(df, save_path=None, target_cols=None):

    global label_encoders
    label_encoders = {}  # dictionary to hold encoders for each target column

    if save_path and os.path.exists(save_path):
        print(f"Loading existing preprocessed file: {save_path}")
        df = pd.read_csv(save_path)
    else:
        print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
            return ' '.join(tokens)

        # Separate target columns
        target_df = None
        if target_cols:
            existing_targets = [col for col in target_cols if col in df.columns]
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

        # Process text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[f'processed_{col}'] = df[col].apply(clean_text)

        # Drop original text columns
        df.drop(columns=text_columns, inplace=True)

        # Reattach target columns
        if target_df is not None:
            for col in target_df.columns:
                df[col] = target_df[col]

        # Save only if path is specified
        if save_path:
            df.to_csv(save_path, index=False)

    # Select processed and numerical columns
    processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
    non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

    # Join processed text columns into one string
    X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

    # Combine with numerical columns if any
    X_numeric = df[non_text_cols].values if non_text_cols else None
    if X_numeric is not None and len(X_numeric) > 0:
        X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
    else:
        X = X_text.tolist()

    # Encode multiple target columns
    Y_dict = {}
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    return X, Y_dict

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def roberta_feature_extraction(texts, model_name='distilroberta-base', batch_size=32, pooling='mean'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting RoBERTa embeddings"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        if pooling == 'mean':
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = token_embeddings[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())

    X = np.vstack(all_embeddings)
    return X

def feature_extraction(X_text, method='Distil RoBERT with Word Embeddings', model_dir='model', is_train=True):
    x_file = os.path.join(model_dir, f'X_{method}.pkl')
    model_name = 'distilroberta-base'

    if os.path.exists(x_file):
        print(f"[INFO] Loading cached features: {x_file}")
        X = joblib.load(x_file)
    else:
        print("[INFO] Cached features not found — extracting new ones...")
        X = roberta_feature_extraction(X_text, model_name=model_name, pooling='mean')
        joblib.dump(X, x_file)

    return X

# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        contact = request.form["contact"]
        address = request.form["address"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            message = "Username already exists."
        else:
            cursor.execute("""
                INSERT INTO users (username, email, contact, address, password)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, email, contact, address, password))
            conn.commit()
            message = "Registration successful. You can now log in."

        cursor.close()
        conn.close()
    return render_template("register.html", message=message)

@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            message = "Invalid username or password."

        cursor.close()
        conn.close()
    return render_template("login.html", message=message)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# ======================================================
# PREDICTION ROUTE
# ======================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    result_html = None
    error = None

    if request.method == "POST":
        try:
            file = request.files["csv_file"]
            if file and file.filename.endswith(".csv"):
                df_test1 = pd.read_csv(file)
                df_result = df_test1.copy()
                df_test, _ = preprocess_data(df_test1)
                features_test = feature_extraction(df_test, method='Distil RoBERT with Word Embeddings', is_train=None)
                model_path = f"model/Distil RoBERT-WE_IndustryGroup_LR_model.pkl"
                model = joblib.load(model_path)
                y_pred = model.predict(features_test)
                y_pred = y_pred[:len(df_result)]
                mapped_labels = [labels1[i] for i in y_pred]
                df_result[f'Predicted_IndustryGroup'] = mapped_labels
                result_html = df_result.to_html(classes="table table-bordered table-striped", index=False)
                output_path = os.path.join(MODEL_DIR, "predictions.csv")
                df_result.to_csv(output_path, index=False)
                print(f"[INFO] Predictions saved to {output_path}")
            else:
                error = "Please upload a valid CSV file."
        except Exception as e:
            error = f"Prediction Error: {str(e)}"

    return render_template("predict.html", result=result_html, error=error)

# ======================================================
# MAIN
# ======================================================
path=r"Dataset/gics-map-2023.csv"
df= upload_dataset(path)
X, Y_dict = preprocess_data(df, save_path="model/cleaned_data.csv", target_cols=["IndustryGroup"])
labels_vars = {}  # dictionary to hold the labels
label_encoders_original = label_encoders
for i, (col, le) in enumerate(label_encoders.items(), start=1):
    var_name = f"labels{i}"
    labels_vars[var_name] = list(le.classes_)

for idx, class_name in enumerate(labels_vars["labels1"]):
    print(f"labels{idx}: {class_name}")

labels1 = labels_vars.get("labels1")



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
