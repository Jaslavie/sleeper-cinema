import ast
import torch
import pandas as pd

from graph import Graph

# Masked away from training to prevent cheating
POST_RELEASE = ["$Worldwide", "$Domestic", "Domestic %", "$Foreign", "Foreign %", "Rating", "Vote_Count", "Budget"]

def resolve_device(device: str) -> torch.device:
    """
    Resolve the device to use for the model.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device: {device}")

def load_processed_movie_data(csv_path: str, device: torch.device, drop_postrelease: bool = True):
    df = pd.read_csv(csv_path)
    scalar_cols = ["$Worldwide", "$Domestic", "Domestic %", "$Foreign", "Foreign %", "Year", "Rating", "Vote_Count", "Original_Language", "Budget"]
    
    # enriched pre-release features (keywords/studio/runtime/season) pulled from TMDB
    scalar_cols += ["runtime", "overview_len", "n_keywords", "n_companies", "studio_vol_max", "studio_vol_min",
                    "has_homepage", "summer", "holiday", "is_sequel"]
    list_cols = ["Genres", "Production_Countries", "Production_Companies", "Keywords"]
    
    # mask post-release metrics so the model trains on PRE-RELEASE features only
    # post release metrics like earnings allows the model to cheat by predicting
    # the outcome itself (circular inference)
    if drop_postrelease:
        scalar_cols = [c for c in scalar_cols if c not in POST_RELEASE]

    rows = []
    for _, row in df.iterrows():
        features = [float(row[col]) for col in scalar_cols]
        for col in list_cols:
            features.extend(ast.literal_eval(row[col]))
        rows.append(features)

    # convert rows and labels into tensors
    X = torch.tensor(rows, dtype=torch.float32, device=device)
    label = torch.tensor(df["Success"].astype(int).to_numpy(), dtype=torch.float32, device=device)
    return df, X, label

def load_graph(graph_path: str) -> Graph:
    graph = Graph(0, []) # 0 vertices, 0 edges
    graph.load_graph(graph_path)
    return graph