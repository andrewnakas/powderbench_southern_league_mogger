"""southern-league-mogger: calibrated multi-model snowfall forecasts for PowderBench."""

HORIZONS = (24, 48, 72)
MAX_SNOWFALL_IN = 200.0
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COLS = {0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"}
POWDER_ALERT_INCHES = 6.0
TEAM = "southern-league-mogger"
