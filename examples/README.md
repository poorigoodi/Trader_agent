# Examples

Offline, network-free demos of the core pipeline. Run each from the
**repository root** with the project venv:

```bash
.venv/Scripts/python.exe examples/01_indicators.py
.venv/Scripts/python.exe examples/02_strategy_signal.py
.venv/Scripts/python.exe examples/03_paper_trade.py
```

| Script | What it shows |
|---|---|
| `01_indicators.py` | In-house EMA / RSI / ATR on synthetic OHLCV |
| `02_strategy_signal.py` | Three-gate strategy → LONG / SHORT / NO_TRADE |
| `03_paper_trade.py` | Risk levels, position size, simulated fill + exit |

These do **not** call any exchange. For the live paper loop, use `main.py`
instead (see the root README).
