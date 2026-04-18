import os
import pandas as pd
import numpy as np
import warnings
import time

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

start_time = time.perf_counter()

# =========================
# Paths and settings
# =========================
data_dir = os.path.expanduser("~/Desktop/monthly klines csv/prices_cleaned")
output_dir = os.path.expanduser("~/Desktop/monthly klines csv/arima_order_selection_outputs")
os.makedirs(output_dir, exist_ok=True)

BASE_DATE = pd.Timestamp("2022-04-01")
REBAL_INTERVAL = 7
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

P_VALUES = range(0, 4)   # 0,1,2,3
D_VALUES = range(0, 2)   # 0,1
Q_VALUES = range(0, 4)   # 0,1,2,3

candidate_orders = [(p, d, q) for p in P_VALUES for d in D_VALUES for q in Q_VALUES]

# =========================
# Helper functions
# =========================
def parse_mixed_time(series, base_date):
    """
    Handle mixed time formats row by row:
    - small values: relative seconds from base_date
    - medium values: Unix seconds
    - very large values: Unix milliseconds
    """
    s = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_rel = s.notna() & (s < 10**9)
    mask_sec = s.notna() & (s >= 10**9) & (s < 10**12)
    mask_ms = s.notna() & (s >= 10**12)

    if mask_rel.any():
        parsed.loc[mask_rel] = base_date + pd.to_timedelta(s.loc[mask_rel], unit="s")
    if mask_sec.any():
        parsed.loc[mask_sec] = pd.to_datetime(s.loc[mask_sec], unit="s", errors="coerce")
    if mask_ms.any():
        parsed.loc[mask_ms] = pd.to_datetime(s.loc[mask_ms], unit="ms", errors="coerce")

    return parsed


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "close" not in df.columns or "time" not in df.columns:
        raise ValueError(f"{os.path.basename(file_path)} must contain 'close' and 'time' columns")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["close", "time"]).copy()

    df["parsed_time"] = parse_mixed_time(df["time"], BASE_DATE)
    df = df.dropna(subset=["parsed_time"]).copy()

    df = df.sort_values("parsed_time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["parsed_time"], keep="last")

    df["return_1step"] = df["close"].pct_change()
    df = df.dropna(subset=["return_1step"]).copy()

    return df[["parsed_time", "close", "return_1step"]].reset_index(drop=True)


def split_series(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy().reset_index(drop=True)
    val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test = df.iloc[val_end:].copy().reset_index(drop=True)

    return train, val, test


def returns_to_price_path(start_price, forecast_returns):
    prices = []
    prev_price = float(start_price)

    for r in forecast_returns:
        next_price = prev_price * (1.0 + float(r))
        prices.append(next_price)
        prev_price = next_price

    return np.array(prices)


def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2


def forecast_is_valid(arr, expected_len):
    arr = np.asarray(arr, dtype=float)
    if len(arr) != expected_len:
        return False
    if np.isnan(arr).any():
        return False
    if np.isinf(arr).any():
        return False
    return True


def fit_arima_safe(series_values, order):
    model = ARIMA(
        series_values,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit()
    return fitted


def walk_forward_rebalance_forecasts_fast(pre_block, target_block, order, rebal_interval=7):
    """
    Validation-stage version:
    fit once on pre_block, then update with append(..., refit=False)
    """
    rebalance_forecasts = []
    rebalance_actuals = []
    rebalance_timestamps = []

    history_values = pre_block["return_1step"].to_numpy(dtype=float)
    fitted = fit_arima_safe(history_values, order)

    rebal_indices = list(range(0, len(target_block), rebal_interval))

    for idx in rebal_indices:
        if idx + rebal_interval > len(target_block):
            break

        if idx == 0:
            last_price = pre_block["close"].iloc[-1]
        else:
            last_price = target_block["close"].iloc[idx - 1]

        actual_end_price = target_block["close"].iloc[idx + rebal_interval - 1]
        actual_return = (float(actual_end_price) / float(last_price)) - 1.0

        try:
            forecast_steps = fitted.forecast(steps=rebal_interval)

            if not forecast_is_valid(forecast_steps, rebal_interval):
                raise ValueError("invalid forecast")

            forecast_prices = returns_to_price_path(last_price, forecast_steps)
            forecast_return = (forecast_prices[-1] / float(last_price)) - 1.0

            rebalance_forecasts.append(forecast_return)
            rebalance_actuals.append(actual_return)
            rebalance_timestamps.append(target_block["parsed_time"].iloc[idx])

            new_actual_data = target_block["return_1step"].iloc[idx: idx + rebal_interval].to_numpy(dtype=float)
            fitted = fitted.append(new_actual_data, refit=False)

        except Exception:
            continue

    return rebalance_timestamps, rebalance_forecasts, rebalance_actuals


def tune_orders_on_validation(train, val, candidate_orders, rebal_interval=7):
    best = None
    tuning_log = []

    for order in candidate_orders:
        try:
            ts, val_forecasts, val_actuals = walk_forward_rebalance_forecasts_fast(
                pre_block=train,
                target_block=val,
                order=order,
                rebal_interval=rebal_interval,
            )

            if len(val_forecasts) < 5:
                tuning_log.append((order, "too few valid validation forecasts"))
                continue

            val_mae, val_rmse, val_r2 = calc_metrics(val_actuals, val_forecasts)
            tuning_log.append((order, val_rmse))

            if best is None or val_rmse < best["val_rmse"]:
                best = {
                    "order_used": order,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "num_val_rebalances": len(val_forecasts),
                }

        except Exception as e:
            tuning_log.append((order, str(e)))
            continue

    return best, tuning_log


# =========================
# Main
# =========================
usable_files = []
for file_name in sorted(os.listdir(data_dir)):
    file_path = os.path.join(data_dir, file_name)

    if not os.path.isfile(file_path):
        continue
    if file_name.startswith("."):
        continue
    if file_name.endswith((".py", ".npy", ".txt", ".ipynb", ".xlsx", ".keras", ".csv")):
        continue

    usable_files.append(file_name)

print(f"Found {len(usable_files)} usable files.")
print("Files:", usable_files)
print(f"Trying {len(candidate_orders)} candidate ARIMA orders per coin...")

summary_rows = []
skipped = []

for file_name in usable_files:
    print(f"\nProcessing {file_name} ...")
    file_path = os.path.join(data_dir, file_name)

    try:
        df = load_and_preprocess(file_path)
        print(f"  Usable rows after preprocessing: {len(df)}")

        train, val, test = split_series(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        print(f"  Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

        if len(train) < 30 or len(val) < REBAL_INTERVAL or len(test) < REBAL_INTERVAL:
            skipped.append((file_name, "not enough data after split"))
            print("  Skipped: not enough data")
            continue

        best_result, tuning_log = tune_orders_on_validation(
            train=train,
            val=val,
            candidate_orders=candidate_orders,
            rebal_interval=REBAL_INTERVAL,
        )

        if best_result is None:
            skipped.append((file_name, tuning_log))
            print("  Skipped: no valid ARIMA order found")
            continue

        summary_rows.append({
            "crypto": file_name,
            "order_used": best_result["order_used"],
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "num_val_rebalances": best_result["num_val_rebalances"],
            "val_mae": best_result["val_mae"],
            "val_rmse": best_result["val_rmse"],
            "val_r2": best_result["val_r2"],
        })

        print(f"  Selected order: {best_result['order_used']}")
        print(f"  Validation RMSE: {best_result['val_rmse']:.8f}")

    except Exception as e:
        skipped.append((file_name, str(e)))
        print(f"  Failed: {e}")

summary_df = pd.DataFrame(summary_rows)
if len(summary_df) > 0:
    summary_df = summary_df.sort_values("val_rmse")
    summary_path = os.path.join(output_dir, "arima_selected_orders.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved selected orders to: {summary_path}")

if len(skipped) > 0:
    skipped_df = pd.DataFrame(skipped, columns=["crypto", "reason"])
    skipped_path = os.path.join(output_dir, "arima_order_selection_skipped.csv")
    skipped_df.to_csv(skipped_path, index=False)
    print(f"Saved skipped log to: {skipped_path}")

end_time = time.perf_counter()
print(f"\nOrder selection done. Runtime: {end_time - start_time:.2f} seconds")