import os
import ast
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
selection_dir = os.path.expanduser("~/Desktop/monthly klines csv/arima_order_selection_outputs")
output_dir = os.path.expanduser("~/Desktop/monthly klines csv/arima_rebalance_outputs")
os.makedirs(output_dir, exist_ok=True)

selected_orders_path = os.path.join(selection_dir, "arima_selected_orders.csv")

BASE_DATE = pd.Timestamp("2022-04-01")
REBAL_INTERVAL = 7
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# =========================
# Helper functions
# =========================
def parse_mixed_time(series, base_date):
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


def fit_arima_safe(series_values, order):
    model = ARIMA(
        series_values,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit()
    return fitted


def fit_and_forecast(series_values, steps, order):
    fitted = fit_arima_safe(series_values, order)
    forecast = fitted.forecast(steps=steps)
    return fitted, forecast


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
    arr = np.asarray(arr)
    if len(arr) != expected_len:
        return False
    if np.isnan(arr).any():
        return False
    if np.isinf(arr).any():
        return False
    return True


def walk_forward_rebalance_forecasts_strict(pre_block, target_block, order, rebal_interval=7):
    rebalance_forecasts = []
    rebalance_actuals = []
    rebalance_timestamps = []

    rebal_indices = list(range(0, len(target_block), rebal_interval))

    for idx in rebal_indices:
        if idx + rebal_interval > len(target_block):
            break

        history_returns = pd.concat([
            pre_block["return_1step"],
            target_block["return_1step"].iloc[:idx]
        ], ignore_index=True)

        if idx == 0:
            last_price = pre_block["close"].iloc[-1]
        else:
            last_price = target_block["close"].iloc[idx - 1]

        actual_end_price = target_block["close"].iloc[idx + rebal_interval - 1]
        actual_return = (float(actual_end_price) / float(last_price)) - 1.0

        try:
            _, forecast_steps = fit_and_forecast(history_returns.to_numpy(dtype=float), rebal_interval, order)

            if not forecast_is_valid(forecast_steps, rebal_interval):
                raise ValueError("invalid forecast")

            forecast_prices = returns_to_price_path(last_price, np.asarray(forecast_steps))
            forecast_return = (forecast_prices[-1] / float(last_price)) - 1.0

            rebalance_forecasts.append(forecast_return)
            rebalance_actuals.append(actual_return)
            rebalance_timestamps.append(target_block["parsed_time"].iloc[idx])

        except Exception:
            continue

    return rebalance_timestamps, rebalance_forecasts, rebalance_actuals


# =========================
# Main
# =========================
if not os.path.exists(selected_orders_path):
    raise FileNotFoundError(
        f"Cannot find selected orders file: {selected_orders_path}\nRun Part 1 first."
    )

selected_orders_df = pd.read_csv(selected_orders_path)

summary_rows = []
all_test_forecast_rows = []
all_test_actual_rows = []
skipped = []

print(f"Loaded selected orders from: {selected_orders_path}")
print(f"Number of coins to process: {len(selected_orders_df)}")

for _, row in selected_orders_df.iterrows():
    file_name = row["crypto"]
    order = ast.literal_eval(row["order_used"])
    file_path = os.path.join(data_dir, file_name)

    print(f"\nProcessing {file_name} with selected order {order} ...")

    try:
        df = load_and_preprocess(file_path)
        print(f"  Usable rows after preprocessing: {len(df)}")

        train, val, test = split_series(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        print(f"  Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

        if len(train) < 30 or len(val) < REBAL_INTERVAL or len(test) < REBAL_INTERVAL:
            skipped.append((file_name, "not enough data after split"))
            print("  Skipped: not enough data")
            continue

        pre_test = pd.concat([train, val], ignore_index=True)

        test_timestamps, test_forecasts, test_actuals = walk_forward_rebalance_forecasts_strict(
            pre_block=pre_test,
            target_block=test,
            order=order,
            rebal_interval=REBAL_INTERVAL
        )

        if len(test_forecasts) < 5:
            skipped.append((file_name, "too few valid test rebalances"))
            print("  Skipped: too few valid test rebalances")
            continue

        test_mae, test_rmse, test_r2 = calc_metrics(test_actuals, test_forecasts)

        summary_rows.append({
            "crypto": file_name,
            "order_used": order,
            "num_test_rebalances": len(test_forecasts),
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })

        test_out = pd.DataFrame({
            "timestamp": test_timestamps,
            "crypto": file_name,
            "actual_return_7d": test_actuals,
            "forecast_return_7d": test_forecasts,
        })
        test_out.to_csv(
            os.path.join(output_dir, f"{file_name}_rebalance_7d_test_forecasts.csv"),
            index=False
        )

        all_test_forecast_rows.append(
            pd.DataFrame({
                "timestamp": test_timestamps,
                "crypto": file_name,
                "value": test_forecasts
            })
        )
        all_test_actual_rows.append(
            pd.DataFrame({
                "timestamp": test_timestamps,
                "crypto": file_name,
                "value": test_actuals
            })
        )

        print(f"  Test RMSE: {test_rmse:.8f}")

    except Exception as e:
        skipped.append((file_name, str(e)))
        print(f"  Failed: {e}")

summary_df = pd.DataFrame(summary_rows)
if len(summary_df) > 0:
    summary_df = summary_df.sort_values("test_rmse")
    summary_path = os.path.join(output_dir, "arima_7d_rebalance_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved test summary to: {summary_path}")

if len(all_test_forecast_rows) > 0:
    forecast_long = pd.concat(all_test_forecast_rows, ignore_index=True)
    forecast_matrix_df = forecast_long.pivot(index="timestamp", columns="crypto", values="value")
    forecast_matrix_df = forecast_matrix_df.sort_index().sort_index(axis=1)
    forecast_matrix_df = forecast_matrix_df.dropna(axis=0, how="any")

    forecast_csv = os.path.join(output_dir, "ARIMA_test_7d_rebalance_forecast_matrix.csv")
    forecast_npy = os.path.join(output_dir, "ARIMA_test_7d_rebalance_forecast_matrix.npy")

    forecast_matrix_df.to_csv(forecast_csv)
    np.save(forecast_npy, forecast_matrix_df.to_numpy())

    print(f"Saved forecast matrix to: {forecast_csv}")
    print(f"Forecast matrix shape: {forecast_matrix_df.shape}")

if len(all_test_actual_rows) > 0:
    actual_long = pd.concat(all_test_actual_rows, ignore_index=True)
    actual_matrix_df = actual_long.pivot(index="timestamp", columns="crypto", values="value")
    actual_matrix_df = actual_matrix_df.sort_index().sort_index(axis=1)
    actual_matrix_df = actual_matrix_df.dropna(axis=0, how="any")

    if 'forecast_matrix_df' in locals():
        common_idx = forecast_matrix_df.index.intersection(actual_matrix_df.index)
        common_cols = forecast_matrix_df.columns.intersection(actual_matrix_df.columns)
        forecast_matrix_df = forecast_matrix_df.loc[common_idx, common_cols]
        actual_matrix_df = actual_matrix_df.loc[common_idx, common_cols]

        forecast_csv = os.path.join(output_dir, "ARIMA_test_7d_rebalance_forecast_matrix.csv")
        forecast_npy = os.path.join(output_dir, "ARIMA_test_7d_rebalance_forecast_matrix.npy")
        forecast_matrix_df.to_csv(forecast_csv)
        np.save(forecast_npy, forecast_matrix_df.to_numpy())

    actual_csv = os.path.join(output_dir, "ARIMA_test_7d_rebalance_actual_matrix.csv")
    actual_npy = os.path.join(output_dir, "ARIMA_test_7d_rebalance_actual_matrix.npy")

    actual_matrix_df.to_csv(actual_csv)
    np.save(actual_npy, actual_matrix_df.to_numpy())

    print(f"Saved actual matrix to: {actual_csv}")
    print(f"Actual matrix shape: {actual_matrix_df.shape}")

if len(skipped) > 0:
    skipped_df = pd.DataFrame(skipped, columns=["crypto", "reason"])
    skipped_path = os.path.join(output_dir, "arima_7d_rebalance_test_skipped.csv")
    skipped_df.to_csv(skipped_path, index=False)
    print(f"Saved skipped log to: {skipped_path}")

end_time = time.perf_counter()
print(f"\nStrict 7-day rebalance test done. Runtime: {end_time - start_time:.2f} seconds")