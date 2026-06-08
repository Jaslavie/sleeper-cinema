from omegaconf import DictConfig
import pytest
from hydra import initialize, compose
from src.gnc import GCNEncoder
import torch
import pandas as pd

from src.utils import resolve_device

@pytest.fixture
def encoder():
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="model")

        return GCNEncoder(
            in_features=cfg.model.in_features, 
            hidden_features=cfg.model.hidden_features, 
            out_features=cfg.model.out_features,
            device=resolve_device(cfg.device)
        )

def test_gnc_encoder(encoder):
    # encoder returns valid tensor shape
    # 10 films x 10 features per film
    A = torch.randn(10, 10)
    X = torch.randn(10, 10)
    Z = encoder(A, X)
    assert Z.shape == (10, encoder.out_features)

def test_expected_movie_profile_similarity():
    df = pd.read_csv("enhanced_box_office_data(2000-2024).csv").set_index("Release Group")
    genres = lambda title: set(df.loc[title, "Genres"].split(", "))

    # select and encode movies
    titles = ["Mission: Impossible II", "The Perfect Storm", "What Women Want"]
    # TODO: build A and X from graph input
    # TODO: encode movies

    # check that mission impossible and perfect storm more similar to what women want
    sim_mission_perfect = cosine_similarity(Z_mission_impossible, Z_perfect_storm)
    sim_mission_women = cosine_similarity(Z_mission_impossible, Z_women)

    assert sim_mission_perfect > sim_mission_women