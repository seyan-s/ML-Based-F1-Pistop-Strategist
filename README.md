
# F1 Pitstop Strategy Simulator

This project analyzes and simulates Formula 1 race strategies using [FastF1](https://theoehrly.github.io/Fast-F1/) telemetry data. It uses pre-trained machine learning models to recommend pit stop strategies and visualize them in comparison with actual race data.

## Features

- Predicts optimal pit stop strategy (1 or 2 stops).
- Predicts lap numbers for each pit stop.
- Compares actual vs. predicted strategy lap times.
- Visualizes race telemetry on track maps.
- Animates actual vs. predicted driver paths.

## Project Structure

```

F1-Pitstop/
├── cache/                     # FastF1 cache directory
├── models/
│   ├── pitstop\_classifier.pkl
│   ├── pitlap1\_regressor.pkl
│   └── pitlap2\_regressor.pkl
├── main.py                    # Main script (the one you provided)
├── requirements.txt
└── README.md

````

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Example `requirements.txt`:

```text
fastf1
pandas
matplotlib
numpy
joblib
```

## How to Use

1. Place your trained models (`pitstop_classifier.pkl`, `pitlap1_regressor.pkl`, `pitlap2_regressor.pkl`) in a `models/` directory.
2. Run the script:

```bash
python main.py
```

3. The output includes:

   * Recommended number of stops and pit laps.
   * Lap time comparison charts.
   * Track map visualization with pit stops.
   * Animated movement of the car.

## Notes

* Make sure to **enable the FastF1 cache** to speed up data loading and avoid hitting API limits.
* This version uses Charles Leclerc (`LEC`) at Silverstone 2023 as an example.

## Troubleshooting

### GitHub Push Issues

If you're pushing this project to GitHub:

* Avoid committing large files (like `.ff1pkl`, `.sqlite`, `.pkl`) directly to Git.
* Use `.gitignore` to exclude cache and model files:

```text
cache/
*.ff1pkl
*.sqlite
*.pkl
```

* Optionally use [Git LFS](https://git-lfs.github.com/) for handling large model files.

---

## License

MIT License. Use and modify freely.

```

