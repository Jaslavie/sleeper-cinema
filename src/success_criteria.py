"""
Sleepers are considered films that outperform their peers in a given year. Sleepers must:
    1. High earning: Have a budget-adjusted gross (gross - budget) that is in the top 10% of its peers
    2. 

"""
import numpy as np
import pandas as pd

SUCCESS_QUANTILE = 0.90

def compute_success(df, graph):
    box = pd.read_csv("enhanced_box_office_data(2000-2024).csv")[["Release Group", "$Worldwide"]]
    tmdb = pd.read_csv("tmdb_5000_movies.csv")
    
    # Get the budget of released films and merge with the box office data
    tmdb = tmdb[tmdb["status"] == "Released"][["title", "budget"]].rename(
        columns={"title": "Release Group", "budget": "Budget"})
    raw = box.merge(tmdb, on="Release Group").drop_duplicates("Release Group")
    raw = raw[(raw["$Worldwide"] > 0) & (raw["Budget"] > 0)]
    m = df[["Release Group"]].merge(raw, on="Release Group", how="left")

    # Compute performance of each film
    perf = np.log(m["$Worldwide"] / m["Budget"]).to_numpy()
    perf = np.where(np.isnan(perf), np.nanmedian(perf), perf)
    
    # The residual is the difference between the film's performance and the mean performance of its peers
    # Default to the median performance if the film has no peers
    resid = np.array([
        perf[i] - perf[[j for j in graph.get_neighbors(i) if j != i] or [i]].mean()
        for i in range(len(df))
    ])

    # Return success label if film performs above 90% of its peers
    return (resid >= np.quantile(resid, SUCCESS_QUANTILE)).astype(int)
