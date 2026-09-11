# Raw data (git-ignored)

Drop the two Kaggle CSVs here. They are **not** committed.

Source: [Hourly energy demand, generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (CC0).

| File | Expected shape |
|---|---|
| `energy_dataset.csv` | 35,064 × 29 |
| `weather_features.csv` | 178,396 × 17 |

```bash
# from repo root, with kaggle CLI configured
kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather -p data/raw --unzip
```
