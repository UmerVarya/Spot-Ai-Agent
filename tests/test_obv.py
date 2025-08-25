import pandas as pd
from trade_utils import OnBalanceVolumeIndicator

def test_on_balance_volume_indicator():
    df = pd.DataFrame({
        "close": [10, 11, 10, 12, 12],
        "volume": [100, 150, 200, 100, 120],
    })
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    assert obv.tolist() == [0, 150, -50, 50, 50]
