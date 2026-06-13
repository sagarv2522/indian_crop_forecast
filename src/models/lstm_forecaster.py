"""PyTorch LSTM forecaster for crop price sequences."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency issue
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    _TORCH_IMPORT_ERROR = exc


ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must use a DatetimeIndex")
    return series.sort_index().asfreq("D").ffill().astype(float)


class SequenceDataset(Dataset):
    def __init__(self, values: np.ndarray, lookback: int = 30):
        self.values = values.astype(np.float32)
        self.lookback = lookback

    def __len__(self) -> int:
        return max(0, len(self.values) - self.lookback)

    def __getitem__(self, idx: int):
        x = self.values[idx : idx + self.lookback]
        y = self.values[idx + self.lookback]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


class PriceLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class LSTMPriceForecaster:
    crop: str = "unknown"
    state: str = "unknown"
    lookback: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    max_samples: int = 10000
    device: str = "cpu"

    model: Optional[PriceLSTM] = field(init=False, default=None)
    scaler: Optional[StandardScaler] = field(init=False, default=None)
    history_: Optional[pd.Series] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover - dependency issue
            raise ImportError(
                "torch is not installed. Install requirements.txt to use LSTMPriceForecaster."
            ) from _TORCH_IMPORT_ERROR
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_values(self, series: pd.Series) -> np.ndarray:
        series = _ensure_series(series)
        values = series.values.reshape(-1, 1)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(values).astype(np.float32).ravel()
        return scaled

    def fit(self, series: pd.Series) -> "LSTMPriceForecaster":
        series = _ensure_series(series)
        scaled = self._prepare_values(series)

        # Adjust lookback for short series
        # Need at least lookback + 1 samples to create at least 1 training sequence
        effective_lookback = min(self.lookback, max(1, len(scaled) - 2))
        
        if len(scaled) <= effective_lookback + 1:
            raise ValueError(f"series too short: {len(scaled)} samples, need >{effective_lookback + 1}")

        if len(scaled) > self.max_samples:
            scaled = scaled[-self.max_samples :]
            series = series.iloc[-self.max_samples :]

        dataset = SequenceDataset(scaled, lookback=effective_lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = PriceLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        # Store effective lookback for later use in forecast
        self._effective_lookback = effective_lookback


        self.model.train()
        for _ in range(self.epochs):
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                optimizer.zero_grad()
                pred = self.model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

        self.history_ = series
        return self

    def forecast(self, steps: int = 30, history: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted yet.")

        if history is None:
            if self.history_ is None:
                raise RuntimeError("No history available. Pass a series or call fit() first.")
            history = self.history_

        history = _ensure_series(history)
        scaled = self.scaler.transform(history.values.reshape(-1, 1)).astype(np.float32).ravel().tolist()
        values = history.values.astype(float).tolist()
        last_date = history.index[-1]
        rows = []
        
        # Use effective lookback from fit, fallback to stored lookback if not available
        lookback = getattr(self, '_effective_lookback', self.lookback)

        self.model.eval()
        with torch.no_grad():
            for step in range(1, steps + 1):
                # Use only the minimum of available history and lookback window
                window_size = min(lookback, len(scaled))
                x = torch.tensor(scaled[-window_size:]).view(1, window_size, 1).to(self.device)
                pred_scaled = self.model(x).item()
                pred = float(self.scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32))[0, 0])
                scaled.append(float(pred_scaled))
                values.append(pred)
                next_date = last_date + pd.Timedelta(days=step)
                rows.append({"date": next_date, "forecast": round(pred, 2), "model": "lstm"})

        return pd.DataFrame(rows)

    def save(self, path: Optional[str] = None) -> str:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted yet.")
        path = path or str(ARTIFACT_DIR / f"lstm_{self.crop.lower().replace(' ', '_')}_{self.state.lower().replace(' ', '_')}.pt")
        payload = {
            "crop": self.crop,
            "state": self.state,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_samples": self.max_samples,
            "device": self.device,
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "history": self.history_,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str) -> "LSTMPriceForecaster":
        if torch is None:  # pragma: no cover - dependency issue
            raise ImportError("torch is not installed.") from _TORCH_IMPORT_ERROR
        payload = torch.load(path, map_location="cpu")
        obj = cls(
            crop=payload["crop"],
            state=payload["state"],
            lookback=payload["lookback"],
            hidden_size=payload["hidden_size"],
            num_layers=payload["num_layers"],
            dropout=payload["dropout"],
            lr=payload["lr"],
            batch_size=payload["batch_size"],
            epochs=payload["epochs"],
            max_samples=payload["max_samples"],
            device=payload["device"],
        )
        obj.model = PriceLSTM(
            input_size=1,
            hidden_size=obj.hidden_size,
            num_layers=obj.num_layers,
            dropout=obj.dropout,
        ).to(obj.device)
        obj.model.load_state_dict(payload["state_dict"])
        obj.scaler = payload["scaler"]
        obj.history_ = payload.get("history")
        obj.model.eval()
        return obj
