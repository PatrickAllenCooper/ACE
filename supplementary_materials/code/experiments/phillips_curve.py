#!/usr/bin/env python3
"""
US Phillips Curve Experiment

Active retrospective learning on static economic data from FRED:
- UNRATE: Unemployment Rate
- FEDFUNDS: Federal Funds Rate  
- MICH: Inflation Expectations
- CPILFESL: Core CPI (target)

The experimentalist learns to selectively reveal historical regimes
(e.g., Great Inflation 1970s vs Great Moderation 1990s) as natural interventions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import logging
import argparse
from datetime import datetime
import pandas as pd

# Optional: Use pandas_datareader for FRED data
try:
    import pandas_datareader as pdr
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False


class PhillipsCurveOracle:
    """
    Oracle containing historical economic data.
    
    Acts as a deterministic environment where "interventions" are
    selective revelation of historical regimes.
    """
    
    def __init__(self, data_path: str = None):
        self.nodes = ["UNRATE", "FEDFUNDS", "MICH", "CPILFESL"]
        self.target = "CPILFESL"
        
        # Define historical regimes as "natural interventions"
        self.regimes = {
            "great_inflation": (1970, 1982),    # High volatility
            "volcker_disinflation": (1979, 1983),
            "great_moderation": (1985, 2007),   # Low volatility
            "great_recession": (2007, 2012),
            "post_crisis": (2012, 2020),
            "covid": (2020, 2023)
        }
        
        if data_path and os.path.exists(data_path):
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif HAS_DATAREADER:
            self._fetch_fred_data()
        else:
            self._generate_synthetic_data()
            
        self._preprocess()
    
    def _fetch_fred_data(self):
        """Fetch real data from FRED."""
        logging.info("Fetching data from FRED...")
        try:
            self.data = pdr.DataReader(
                self.nodes, 'fred', 
                start='1960-01-01', end='2023-12-31'
            )
            self.data = self.data.dropna()
            logging.info(f"Loaded {len(self.data)} records from FRED")
        except Exception as e:
            logging.warning(f"FRED fetch failed: {e}, using synthetic data")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic Phillips Curve data for testing."""
        logging.info("Generating synthetic Phillips Curve data...")
        
        n_years = 60
        n_months = n_years * 12
        dates = pd.date_range('1965-01-01', periods=n_months, freq='M')
        
        np.random.seed(42)
        
        # Unemployment: mean-reverting with regime shifts
        unrate = np.zeros(n_months)
        unrate[0] = 5.0
        for t in range(1, n_months):
            regime_shift = 2.0 if 120 < t < 180 else 0  # 1970s spike
            unrate[t] = unrate[t-1] + 0.1 * (5.5 + regime_shift - unrate[t-1]) + np.random.randn() * 0.3
        unrate = np.clip(unrate, 2, 12)
        
        # Fed Funds: policy response to inflation
        fedfunds = np.zeros(n_months)
        fedfunds[0] = 4.0
        
        # Inflation expectations: adaptive
        mich = np.zeros(n_months)
        mich[0] = 3.0
        
        # Core CPI: Phillips curve relationship
        cpilfesl = np.zeros(n_months)
        cpilfesl[0] = 2.5
        
        for t in range(1, n_months):
            # Phillips curve: inflation inversely related to unemployment
            phillips = -0.3 * (unrate[t] - 5.5)
            expectations = 0.5 * mich[t-1]
            policy = -0.1 * fedfunds[t-1]
            
            cpilfesl[t] = cpilfesl[t-1] + phillips + expectations + policy + np.random.randn() * 0.2
            cpilfesl[t] = np.clip(cpilfesl[t], 0, 15)
            
            # Update expectations (adaptive)
            mich[t] = 0.7 * mich[t-1] + 0.3 * cpilfesl[t] + np.random.randn() * 0.1
            
            # Policy response (Taylor-like rule)
            fedfunds[t] = 2.0 + 1.5 * (cpilfesl[t] - 2.0) + 0.5 * (5.5 - unrate[t])
            fedfunds[t] = np.clip(fedfunds[t], 0, 20) + np.random.randn() * 0.2
        
        self.data = pd.DataFrame({
            "UNRATE": unrate,
            "FEDFUNDS": fedfunds,
            "MICH": mich,
            "CPILFESL": cpilfesl
        }, index=dates)
        
    def _preprocess(self):
        """Normalize and prepare data."""
        self.data = self.data.dropna()
        self.data_normalized = (self.data - self.data.mean()) / self.data.std()
        self.years = self.data.index.year
        
    def get_regime_data(self, regime: str) -> pd.DataFrame:
        """Get data from a specific historical regime."""
        if regime not in self.regimes:
            raise ValueError(f"Unknown regime: {regime}")
        start, end = self.regimes[regime]
        mask = (self.years >= start) & (self.years < end)
        return self.data_normalized[mask]
    
    def generate(self, n_samples: int, regime: str = None) -> Dict[str, torch.Tensor]:
        """Sample data, optionally from a specific regime."""
        if regime:
            df = self.get_regime_data(regime)
        else:
            df = self.data_normalized
            
        if len(df) < n_samples:
            indices = np.random.choice(len(df), n_samples, replace=True)
        else:
            indices = np.random.choice(len(df), n_samples, replace=False)
            
        return {col: torch.tensor(df.iloc[indices][col].values, dtype=torch.float32)
                for col in self.nodes}
    
    def get_parents(self, node: str) -> List[str]:
        """Hypothesized causal structure for Phillips Curve."""
        graph = {
            "UNRATE": [],  # Exogenous (labor market)
            "FEDFUNDS": ["CPILFESL", "UNRATE"],  # Policy response
            "MICH": ["CPILFESL"],  # Adaptive expectations
            "CPILFESL": ["UNRATE", "MICH", "FEDFUNDS"]  # Phillips + expectations + policy
        }
        return graph.get(node, [])


class PhillipsCurveLearner(nn.Module):
    """Learner for the Phillips Curve mechanism."""
    
    def __init__(self, input_nodes: List[str], hidden_dim: int = 32):
        super().__init__()
        self.input_nodes = input_nodes
        self.n_inputs = len(input_nodes)
        
        self.mechanism = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.stack([data[n] for n in self.input_nodes], dim=1)
        return self.mechanism(x).squeeze(-1)


def run_phillips_experiment(n_episodes: int = 50, steps_per_episode: int = 20,
                            output_dir: str = "results"):
    """Run the Phillips Curve retrospective learning experiment."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"phillips_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Setup
    oracle = PhillipsCurveOracle()
    input_nodes = ["UNRATE", "MICH", "FEDFUNDS"]
    
    records = []
    regime_order = list(oracle.regimes.keys())
    
    logging.info(f"Available regimes: {regime_order}")
    logging.info(f"Data shape: {oracle.data.shape}")
    
    for episode in range(n_episodes):
        # Fresh learner each episode
        learner = PhillipsCurveLearner(input_nodes)
        optimizer = optim.Adam(learner.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        for step in range(steps_per_episode):
            # Active regime selection strategy
            # Early: expose to diverse regimes
            # Later: focus on high-information regimes
            if step < 5:
                regime = None  # Mixed sample
            elif step < 10:
                regime = "great_inflation"  # High volatility
            elif step < 15:
                regime = "great_moderation"  # Low volatility  
            else:
                # Alternate to expose structural breaks
                regime = regime_order[step % len(regime_order)]
            
            # Generate data
            data = oracle.generate(n_samples=50, regime=regime)
            
            # Train
            learner.train()
            for _ in range(10):
                optimizer.zero_grad()
                pred = learner(data)
                loss = loss_fn(pred, data["CPILFESL"])
                loss.backward()
                optimizer.step()
            
            # Evaluate on held-out mixed data
            learner.eval()
            with torch.no_grad():
                eval_data = oracle.generate(n_samples=100, regime=None)
                eval_pred = learner(eval_data)
                eval_loss = loss_fn(eval_pred, eval_data["CPILFESL"]).item()
            
            records.append({
                "episode": episode,
                "step": step,
                "regime": regime or "mixed",
                "train_loss": loss.item(),
                "eval_loss": eval_loss
            })
        
        if episode % 10 == 0:
            logging.info(f"Episode {episode}: Eval Loss={eval_loss:.4f}")
    
    # Save results
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, "phillips_results.csv"), index=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Learning curves
    ax = axes[0]
    for ep in range(0, n_episodes, 10):
        ep_data = df[df["episode"] == ep]
        ax.plot(ep_data["step"], ep_data["eval_loss"], label=f"Ep {ep}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Phillips Curve: Learning via Regime Selection")
    ax.legend()
    
    # Regime effectiveness
    ax = axes[1]
    regime_losses = df.groupby("regime")["eval_loss"].mean().sort_values()
    ax.barh(regime_losses.index, regime_losses.values)
    ax.set_xlabel("Mean Eval Loss")
    ax.set_title("Information Value by Regime")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "phillips_learning.png"), dpi=150)
    plt.close()
    
    # Save oracle data for reference
    oracle.data.to_csv(os.path.join(run_dir, "economic_data.csv"))
    
    logging.info(f"Results saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    
    run_phillips_experiment(args.episodes, args.steps, args.output)
