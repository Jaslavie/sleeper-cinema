# Sleeper Cinema

Inspired by the recent successes of low-budget/non-hollywood films "Backrooms" and "Obsession", we wanted to investigate Out-of-distribution box office hits and wether or not there were unseen patterns in their success. 

We detect "sleeper" movies that end up topping box-offices with a two-step approach:

1. Get the top-performing films by box office and **isolate anomalies** (Generative Semi-supervised Graph Anomaly Detection Approach)
2. Run a simple algorithm on top of those anomalies to uncover hidden trends in success

Definitionally, sleepers are not necessarily extra-ordinary films with extremely low budgets, rather they can exist as unremarkable/ordinary films on paper and still outperform its original expectations

## Dataset

We keep a focused set of columns from the two source datasets, combined into a single dataset:

**Dataset 1: `enhanced_box_office_data(2000-2024).csv`**

- `Rank`: box-office rank within the dataset.
- `Release Group`: movie title used as the film identifier.
- `$Worldwide`: total worldwide box-office gross.
- `$Domestic`: domestic box-office gross.
- `Domestic %`: domestic share of worldwide gross.
- `$Foreign`: international box-office gross.
- `Foreign %`: international share of worldwide gross.
- `Year`: release year.
- `Genres`: movie genres for profile similarity.
- `Rating`: audience rating score.
- `Production_Countries`: production country metadata.

**Dataset 2: `tmdb_5000_movies.csv`**

- `budget`: reported production budget.
- `production_companies`: production company metadata.

## Installation

Create a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable development with test dependencies from `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

## Architecture

Beforehand, we will construct the film attribution graph.

1. **Anomaly detection**: We use a Variational Graph Autoencoder to learn the latent gaussian of each film and reconstructs the graph. Here, the films are the "nodes" while features like genre act as the "edges" connecting the nodes. We detect films that have a high reconstruction error, thus falling out of the success manifold.
2. **Pattern recognition**: We use a simple HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) Clustering approach to extract trend clusters from the VGAE latent embeddings.

## References

Related material, not necessarily implemented into this architecture.

- [Credal Graph Neural Networks](https://arxiv.org/abs/2512.02722)
- [Generative Semi-supervised Graph Anomaly Detection]()

