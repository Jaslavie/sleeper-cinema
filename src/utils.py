import ast
import torch
import pandas as pd

from graph import Graph

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

def load_processed_movie_data(csv_path: str, device: torch.device):
    df = pd.read_csv(csv_path)
    scalar_cols = ["$Worldwide", "$Domestic", "Domestic %", "$Foreign", "Foreign %", "Year", "Rating", "Vote_Count", "Original_Language", "Budget"]
    list_cols = ["Genres", "Production_Countries", "Production_Companies"]

    rows = []
    for _, row in df.iterrows():
        features = [float(row[col]) for col in scalar_cols]
        for col in list_cols:
            features.extend(ast.literal_eval(row[col]))
        rows.append(features)

    X = torch.tensor(rows, dtype=torch.float32, device=device)
    normal_mask = torch.tensor(~df["Success"].astype(bool).to_numpy(), dtype=torch.bool, device=device)
    return df, X, normal_mask

def load_graph(graph_path: str) -> Graph:
    graph = Graph(0, []) # 0 vertices, 0 edges
    graph.load_graph(graph_path)
    return graph