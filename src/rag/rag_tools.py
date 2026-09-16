import pandas as pd
import joblib
import json
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = "vectorstore/chroma_db"
DATA_PATH = "data/ml_feature_ready.csv"
MODEL_PATH = "models/random_forest_tuned_final.pkl"

@tool
def query_technical_manuals(query: str) -> str:
    """Useful for searching machine repair guides, error code descriptions, and maintenance protocols."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant maintenance documentation found."
    return "\n\n".join([doc.page_content for doc in results])

@tool
def get_machine_telemetry_status(udi: int) -> str:
    """Useful for querying the latest telemetry, sensor readings, and failure probability for a machine by its UDI ID."""
    df = pd.read_csv(DATA_PATH)
    record = df[df["UDI"] == udi] if "UDI" in df.columns else df.iloc[[udi]]
    
    if record.empty:
        return f"No telemetry record found for Machine ID / UDI {udi}."
    
    row = record.iloc[0].to_dict()
    
    # Predict failure probability if model exists
    try:
        model = joblib.load(MODEL_PATH)
        # Load metadata feature ordering
        with open("reports/classical_ml_final_model_metadata.json", "r") as f:
            expected_feats = json.load(f).get("feature_names", [])
            
        from src.xai_utils import prepare_features
        X_proc = prepare_features(record, expected_feats)
        risk = float(model.predict_proba(X_proc.values)[:, 1][0])
        row["Predicted_Failure_Risk"] = f"{risk * 100:.2f}%"
    except Exception:
        row["Predicted_Failure_Risk"] = "N/A"
        
    return json.dumps(row, indent=2, default=str)