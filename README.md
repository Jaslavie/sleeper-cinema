# Sleeper Cinema: Predicting Box-Office Outliers with Markov Fields

> *Can we systematically detect underdogs in films?*

Inspired by the recent successes of low-budget/non-hollywood films "Backrooms" and "Obsession", we wanted to investigate Out-of-distribution box office hits and wether or not there were unseen patterns in their success. 

We detect "sleeper" movies that end up topping box-offices with a two-step approach:

1. Get the top-performing films and **isolate anomalies** (Films that outperform expectations given its attributes)
2. Run a simple algorithm on top of those anomalies to uncover hidden trends in said success

Definitionally, sleepers are not necessarily extra-ordinary films with extremely low budgets, rather they can exist as unremarkable/ordinary films on paper and still outperform its original expectations.

## Motivation

Every year, a handful of films defy expectations. Despite limited budgets and the absence of major studio backing, these movies outperform industry expectations and achieve remarkable box office success. These “sleeper” hits challenge conventional prediction models, which are typically trained on patterns observed in mainstream releases. When a film succeeds for reasons that fall outside those patterns, traditional approaches often fail to recognize it. We investigate whether graph-based learning can fill that gap by leveraging structural relationships between films to detect and explain anomalous box office performance. 

**Markov Random Fields** are particularly powerful for this compared to other approaches, and anomalies are fundmentally influenced by the groups they're in. For example, a hollywood film making $90M on opening night may be quite normal, however it is [record-shattering](https://deadline.com/2026/06/box-office-backrooms-a24-record-1236942792/) for smaller indie films.

## Dataset

We keep a focused set of columns from the two source datasets, combined into a single dataset:

**Dataset 1: `enhanced_box_office_data(2000-2024).csv*`*

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

To run the script to post train the Gradient Boost Model on our selected dataset and run inference: 

```
python src/model.py
```

Returns the ranked anomalies in `crf_ranked.csv` file

## Architecture

The intuition is that the graph is primarily useful for *knowledge representation.*

A simple supervised approach to learning patterns in a film's "DNA" that historically have been associated with outsized success. 

### Anomaly detection

First, we attempt to weed out anomalies from the graph. We do this by scoring each film (node) in isolation, then updating those scores via mean field inference based on its neighbors.

1. **Graph construction**: We connect movies whose genre profiles overlap by at least 60% Jaccard similarity. Each movie is a node, and its first-degree neighbors are its closest peer films.
2. **Success labeling**: Sleeper successes are define based on how competitive its gross/budget ratio is compared to peers
3. **Supervised learning**: We train a gradient boosted tree on pre-release featuers . This gives each film an individual sleeper score before considering its neighbors.
4. **Graph smoothing**: We run mean-field inference over the graph so similar films can influence each other's sleeper probability. Known training labels stay fixed, and validation films receive final ranked scores.

Then, we use a simple algorithm to reveal trends in these movies.

## Baselines

We compare our model against two graph anomaly-detection baselines and two non-graph references. All metrics are on the same temporal 80/20 hold-out (validation base rate 0.148, the AUROC chance level is 0.50 and the AUPRC chance level equals the base rate).


| Method                                                 | Uses graph | AUROC     | AUPRC     | P@10     |
| ------------------------------------------------------ | ---------- | --------- | --------- | -------- |
| Random ranking (chance)                                | –          | 0.500     | 0.148     | 0.148    |
| Logistic regression (pre-release features)             | ✗          | 0.640     | –         | –        |
| **GGAD** (one-class, GCN encoder + synthetic outliers) | ✓          | 0.590     | –         | –        |
| **GNN** (GCN embedding added to supervised features)   | ✓          | 0.620     | –         | –        |
| GBM unary only (β = 0 ablation)                        | ✗          | 0.723     | 0.345     | –        |
| **MRF + supervised unary (ours)**                      | ✓          | **0.724** | **0.329** | **0.50** |


**Graphical Neural Network (GNN).** A two-layer GCN encodes each film from the adjacency matrix and feature matrix into a semantic embedding. The embedding is smoothed to mirror its genre neighbors in the graph.

**GGAD (Generative Semi-supervised Graph Anomaly Detection).** GGAD trains without labeled anomalies by synthesizing fake outlier nodes from normal nodes and learning to separate them. It achieved low success (0.59 AUROC) since the architecture assumes that anomalies are fundamentally different in the *feature space*, however anomalies do not deviate at all in this (i.e. they can look quite normal to other normal-performing films) but the anomaly is in the *post-release revenue signal*. 

**Takeaway.** Pure graph anomaly detection (GGAD, GNN) underperforms even a plain logistic baseline. What works is training the model with labled anomalies so that it learns the characteristics and patterns of sleeper films. The graph offers the structural foundation from which an CRF model can guess the anomaly score based on a film's neighbors.

## References

Related material, not necessarily implemented into this architecture.

- [Credal Graph Neural Networks](https://arxiv.org/abs/2512.02722)
- [Generative Semi-supervised Graph Anomaly Detection]()

