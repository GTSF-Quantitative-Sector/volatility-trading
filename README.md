# volatility-trading

Research project for a VIX term-structure signal that adjusts SPY exposure. The current code downloads the core daily series from the [Massive API](https://massive.com/docs). Strategy context for contributors and agents is in [AGENTS.md](AGENTS.md).

## API key

Create a local `.env` file. It is gitignored.

```bash
cp .env.example .env
```

Put the key on the first line:

```bash
MASSIVE_API_KEY=your_key_here
```

`POLYGON_API_KEY` still works as a fallback. Do not commit the key or the downloaded files.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

By default this writes daily bars from `2023-02-14` through today's New York date:

- `data/raw/SPY.csv`
- `data/raw/I_VIX.csv`
- `data/raw/I_VIX3M.csv`

`2023-02-14` is the start of Massive index aggregate history. The strategy outline studies the curve from 2010 onward, so that longer sample needs another source later. SPY uses split-adjusted prices, not total return.

Override the window or destination when needed:

```bash
python main.py --start 2024-01-01 --end 2024-12-31 --output-dir data/raw
```

`polygon-api-demo.ipynb` remains a reference for the broader Massive endpoints. It is not the ingestion entry point.
