import argparse
import yaml
from pathlib import Path
from datetime import datetime
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


from src.swingbt.utils.data_loader import load_prices
from src.swingbt.backtest.run_backtest import run_backtest




HORIZON_KEYS = {
"daily": (1, 1),
"weekly": (5, 5),
"monthly": (21, 21),
}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=["daily","weekly","monthly"], default="weekly")
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()


    cfg = yaml.safe_load(Path(args.config).read_text())
    lookahead, rebalance = HORIZON_KEYS[args.horizon]
    cost_bps = cfg['horizons'][args.horizon]['cost_bps_per_trade']


    prices = load_prices(cfg['universe']['path'], cfg['universe']['source'])


    res = run_backtest(
        prices=prices,
        benchmark=cfg['universe']['benchmark'],
        lookahead_days=lookahead,
        rebalance_days=rebalance,
        cost_bps=cost_bps,
        train_years=cfg['validation']['train_years'],
        test_months=cfg['validation']['test_months'],
        embargo_months=cfg['validation']['embargo_months'],
        top_q=cfg['portfolio']['top_quantile'],
        sector_caps=cfg['portfolio']['sector_caps'],
    )


    out_dir = Path(cfg['io']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    (out_dir / f"{args.horizon}_returns.csv").write_text(res['returns_net'].to_csv())


    print("==== RESULTS ====")
    print(f"IC mean: {res['ic_mean']:.4f} | IC IR: {res['ic_ir']:.2f}")
    s = res['stats']
    print(f"Sharpe: {s['sharpe']:.2f} | AnnRet: {s['ann_return']:.2%} | AnnVol: {s['ann_vol']:.2%} | MaxDD: {s['max_dd']:.2%}")




if __name__ == "__main__":
    main()