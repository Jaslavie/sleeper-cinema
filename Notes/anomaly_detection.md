# Anomaly detection model

Anomaly detection is built from a **GCN encoder** (deliberately *not* variational) which takes in an adjacency matrix of nodes and their connections and a feature matrix of each movie as input.

Specifically, we re-implement the **lowest-scope faithful version of Generative Semi-supervised Graph Anomaly Detection (GGAD)**: the GCN encoder plus the one component that actually defines the paper — outlier nodes shaped by two priors — with every optional extension stripped out.

Original paper: [https://arxiv.org/html/2402.11887](https://arxiv.org/html/2402.11887)

> The key insight of GGAD is to **generate learnable outlier nodes in the feature representation space** that assimilate anomaly nodes in terms of both local structure affinity and feature representation

## Model selection: dual-headed VGAE vs. GGAD

We considered two graph-based architectures for detecting sleeper films:

- **[Dual-headed VGAE](https://medium.com/data-science/tutorial-on-variational-graph-auto-encoders-da9333281129) (reconstruction-based).** This model encodes every film into a probabilistic latent space (Z_recon) and then tries to rebuild both the graph structure and each film's box-office attribute from that latent code. A film is flagged as anomalous when the model rebuilds it poorly, so the anomaly signal is just the size of the reconstruction error.
  - Weakness: trained on full reconstruction loss (`L_recon + L_KL`). This means as the model gets better, the encoder will learn to reconstruct anomalies as faithfully as normal films. The signal is flattened and the anomalies become invisible (abnormality leakage)
- **GGAD — Generative Semi-supervised Graph Anomaly Detection (Generative, separation based model).** This model does not rebuild the graph at all. Instead it learns what normal films look like, *manufactures* fake anomaly nodes ("outlier nodes") that imitate how real anomalies sit in the graph, and then trains a classifier to draw a sharp boundary between normal films and those fakes.
  - GGAD avoids the reconstruction trap by never relying on reconstruction in the first place. Because it generates explicit pseudo-anomalies and trains a **discriminative** classifier against them, the model is optimized to *separate* anomalies rather than to *reproduce* them, so a well-trained model pushes anomalies further away instead of absorbing them. 
  - **Data alignment**: GGAD also matches our data situation almost perfectly: it is a semi-supervised method designed for the case where we confidently know what "normal" looks like (the large mass of expected-performance films) and have only a handful of confirmed examples of the thing we are hunting (known sleepers like *Backrooms* and *Obsession*). 
  - **Architecturally**: It generates its outliers using two priors that fit our intuition about sleepers — that an anomaly is less similar to its neighbors than a normal film is (asymmetric local affinity), and that an anomaly can still look deceptively ordinary in its raw features (egocentric closeness) — which is exactly what a sleeper is: a film that looks unremarkable on paper yet behaves nothing like its peers at the box office.

## Input data contract

The model is handed a single object holding three arrays. Training reads `Attributes` and `Network` for every node, but only computes loss over the `Label == 0` (normal) nodes.

```text
boxoffice.mat                           # full training data vehicle
├─ Attributes : ndarray       [N × F]   # Feature matrix, where each film N gets feature vector F (X)         
├─ Network    : sparse float  [N × N]   # Adjacency matrix (A)
└─ Label      : ndarray int   [N]       # 0 = normal (trained on), 1 = anomaly/sleeper (eval only)
```

Each array is broken down below.

Example `X` (`Attributes`) feature matrix: shape `N × F`, where each row is one film and each column is one model feature.

```text
node_id  film        log_boxoffice  rating_R  vote_count  foreign_share  year  genre_horror  genre_scifi  genre_music_biopic  lang_en  country_us
0        Backrooms   18.76          1         2800        0.263          2026  1             1            0                   1        1
1        Obsession   18.93          1         4500        0.269          2026  1             0            0                   1        1
2        Scream 7    19.15          1         0           0.414          2026  1             0            0                   1        1
3        Michael     20.57          0         0           0.597          2026  0             0            1                   1        1
```

Example `A` adjacency matrix: shape `N × N`, where `1` means two films are profile-similar neighbors and the diagonal self-loops are always `1`.

```text
             Backrooms  Obsession  Scream 7  Michael
Backrooms    1          1          1         0
Obsession    1          1          1         0
Scream 7     1          1          1         0
Michael      0          0          0         1
```

Example `Label` vector: shape `N`, aligned to the same node order as `X` and `A`.

```text
node_id  film        label
0        Backrooms   1        # anomaly — held out for evaluation
1        Obsession   1        # anomaly
2        Scream 7    0        # normal — used for training set
3        Michael     0        # normal
```

## Anomaly model architecture

This section describes the model assuming the graph has already been built and handed to it. It walks from the structure of that input all the way to the final list of detected sleepers.

**1. The input graph.** The model receives one attributed graph.

- **Nodes:** each node is one film, giving a fixed population of `N` films (determined by dataset)
- **Node attributes:** `X` (feature matrix) has shape `N × F`, with one feature vector per film.
  - Features include log-scaled box office, rating, vote count, foreign share, and year.
  - Categorical context is encoded with multi-hot or one-hot genre, country, and language fields.
- **Adjacency matrix:** `A` has shape `N × N` represent connections between nodes on the graph
  - It is a symmetric binary matrix with self-loops.
  - `A[i, j] = 1` when two films are profile-similar neighbors.
  - This lets each film be judged against similar films rather than the full catalog.
- **Normal-node mask:** a subset of nodes are marked confidently normal (the only labels used).
  - These are films whose box office matches expectation.
  - The adjacency is read at *training* time too, since the affinity prior compares each node to its real neighbors.

**2. Graph encoder.** A GCN-style encoder consumes `A` and `X`

```math
GCN(A, X)
```

- Stacked message-passing layers produce a `d`-dimensional embedding for every film.
- Each embedding combines the film's own features with information from its similar-film neighborhood.
- This is the space where "fits its peers" versus "breaks from its peers" becomes measurable.

**3. Outlier (pseudo-anomaly) generation.** The model synthesizes fake anomaly nodes from labeled-normal embeddings.

- Each generated outlier starts from a learnable transformation of averaged normal-neighbor representations (roughly one outlier per normal node).
- The generator is shaped by two priors. The quality of generated outliers is controlled by:
  - **Egocentric closeness:** Gaussian perturbation keeps the outlier near the normal feature manifold.
    - Ensures that outlier movies still look similar to normal movies so it's not totally random
  - **Asymmetric local affinity:** a margin penalty makes the outlier less similar to its neighbors than real normal nodes are.
    - Ensures that outlier movies do not fit as well to its neighbors as true normal movies do
- Drop either prior and the outliers collapse to random noise, reducing GGAD to a generic one-class classifier — so these stay while variational latents and reconstruction heads do not.

**4. Discriminative one-class head.** A classifier sits on top of the embeddings.

- It assigns high scores to normal films and low scores to generated outliers.
- The end-to-end objective combines:
  - binary cross-entropy separating labeled normals from synthetic outliers;
  - affinity and closeness terms that keep the generated outliers realistic.
- There is no graph reconstruction term, which prevents abnormality leakage.

**5. Detection.** At inference, every unlabeled film receives one anomaly score.

- The trained encoder and classifier score each film.
- Films are sorted by anomaly score.
- The high-anomaly tail becomes the candidate sleeper list.

## Evaluation

We hold out our known sleepers and measure ranking quality with AUROC and AUPRC, and report precision@k on the top-scored films to check how many true sleepers appear at the very top of the list. As a lightweight sanity check we confirm that obvious blockbusters score as normal and that confirmed sleepers land in the high-anomaly tail.

More information of this in the evaluations document.