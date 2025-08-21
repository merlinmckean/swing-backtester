from __future__ import annotations
import numpy as np




def apply_trading_costs(returns: np.ndarray, turnovers: np.ndarray, cost_bps_per_trade: float) -> np.ndarray:
    """Apply linear transaction costs per unit turnover.
    returns: per-period portfolio return array
    turnovers: per-period turnover in [0,1]
    cost_bps_per_trade: e.g., 5 = 5 bps per rebalance
    """
    costs = turnovers * (cost_bps_per_trade / 1e4)
    return returns - costs