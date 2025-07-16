def map_symbol_for_binance(symbol):
    custom_map = {
        "SHIBUSDT": "1000SHIBUSDT",
        "PEPEUSDT": "1000PEPEUSDT",
        "LUNCUSDT": "1000LUNCUSDT",
        # Add others if needed
    }
    return custom_map.get(symbol, symbol)
