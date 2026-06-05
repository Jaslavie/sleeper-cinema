# Sleeper Cinema

Inspired by the recent successes of low-budget/non-hollywood films "Backrooms" and "Obsession", we wanted to investigate Out-of-distribution box office hits and wether or not there were unseen patterns in their success.

We detect "sleeper" movies that end up topping box-offices with a two-step approach:

1. Get the top-performing films by box office and **isolate anomalies**
2. Run a simple algorithm on top of those anomalies to uncover hidden trends in success

### Architecture

Beforehand, we will construct the film attribution graph.

1. **Anomaly detection**: We use a Variational Graph Autoencoder to learn the latent gaussian of each film and reconstructs the graph. Here, the films are the "nodes" while features like genre act as the "edges" connecting the nodes. We detect films that have a high reconstruction error, thus falling out of the success manifold.
2. **Pattern recognition**: We use a simple HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) Clustering approach to extract trend clusters from the VGAE latent embeddings.

### References

Related material, not necessarily implemented into this architecture.

- [Credal Graph Neural Networks](https://arxiv.org/abs/2512.02722)

