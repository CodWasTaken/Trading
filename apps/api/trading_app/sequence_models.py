from __future__ import annotations

import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .modeling import ModelCapabilities
from .research import FeatureRow


class TorchSequenceReturnModel:
    """Lazy PyTorch sequence adapter for LSTM, GRU, TCN, PatchTST, and iTransformer."""

    def __init__(
        self,
        implementation: str,
        feature_names: tuple[str, ...],
        *,
        hyperparameters: dict[str, object] | None = None,
        seed: int = 1729,
        forecast_horizon: int = 5,
    ) -> None:
        if implementation not in {"lstm", "gru", "tcn", "patchtst", "itransformer"}:
            raise ValueError(f"Unsupported sequence model: {implementation}")
        self.hyperparameters = hyperparameters or {}
        self.seed = seed
        self.context_length = int(self.hyperparameters.get("context_length", 20))
        if self.context_length < 3:
            raise ValueError("Sequence context_length must be at least three")
        self.capabilities = ModelCapabilities(
            family="sequence",
            implementation=implementation,
            feature_names=feature_names,
            forecast_horizon=forecast_horizon,
            required_history=self.context_length,
            optional_dependency="torch",
            deterministic=True,
        )
        self.network: Any = None

    def _torch(self) -> Any:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for sequence models") from error
        return torch

    def _build_network(self) -> Any:
        torch = self._torch()
        nn = torch.nn
        implementation = self.capabilities.implementation
        width = len(self.capabilities.feature_names)
        hidden = int(self.hyperparameters.get("hidden_size", 32))
        layers = int(self.hyperparameters.get("layers", 1))
        heads = int(self.hyperparameters.get("heads", 4))
        dropout = float(self.hyperparameters.get("dropout", 0.0))

        if implementation in {"lstm", "gru"}:
            recurrent_class = nn.LSTM if implementation == "lstm" else nn.GRU

            class RecurrentNetwork(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.recurrent = recurrent_class(
                        width,
                        hidden,
                        num_layers=layers,
                        batch_first=True,
                        dropout=dropout if layers > 1 else 0.0,
                    )
                    self.head = nn.Linear(hidden, 1)

                def forward(self, value: Any) -> Any:
                    encoded, _ = self.recurrent(value)
                    return self.head(encoded[:, -1, :]).squeeze(-1)

            return RecurrentNetwork()

        if implementation == "tcn":
            class TemporalConvNetwork(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv1d(width, hidden, kernel_size=3, padding=2, dilation=1),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
                        nn.ReLU(),
                    )
                    self.head = nn.Linear(hidden, 1)

                def forward(self, value: Any) -> Any:
                    encoded = self.net(value.transpose(1, 2))[:, :, : value.shape[1]]
                    return self.head(encoded[:, :, -1]).squeeze(-1)

            return TemporalConvNetwork()

        class TransformerNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Linear(width, hidden)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=max(1, min(heads, hidden)),
                    dim_feedforward=hidden * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.head = nn.Linear(hidden, 1)

            def forward(self, value: Any) -> Any:
                encoded = self.encoder(self.input(value))
                pooled = encoded.mean(dim=1) if implementation == "patchtst" else encoded[:, -1, :]
                return self.head(pooled).squeeze(-1)

        return TransformerNetwork()

    def fit(self, rows: list[FeatureRow]) -> None:
        torch = self._torch()
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        examples = sequence_examples(rows, self.context_length)
        if not examples:
            raise ValueError("Not enough point-in-time rows to build sequence examples")
        self.network = self._build_network()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.hyperparameters.get("learning_rate", 1e-3)),
            weight_decay=float(self.hyperparameters.get("weight_decay", 0.0)),
        )
        loss_function = torch.nn.MSELoss()
        epochs = int(self.hyperparameters.get("epochs", 20))
        batch_size = int(self.hyperparameters.get("batch_size", 64))
        x = torch.tensor([item[0] for item in examples], dtype=torch.float32)
        y = torch.tensor([item[1] for item in examples], dtype=torch.float32)
        self.network.train()
        for _ in range(epochs):
            for start in range(0, len(examples), batch_size):
                batch_x = x[start : start + batch_size]
                batch_y = y[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_function(self.network(batch_x), batch_y)
                loss.backward()
                optimizer.step()

    def predict_context(self, context: tuple[tuple[float, ...], ...]) -> float:
        if self.network is None:
            raise RuntimeError("Model is not fitted")
        if len(context) < self.context_length:
            raise ValueError("Insufficient sequence context")
        torch = self._torch()
        value = torch.tensor([context[-self.context_length :]], dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            return float(self.network(value)[0].item())

    def predict(self, features: tuple[float, ...]) -> float:
        return self.predict_context(tuple(features for _ in range(self.context_length)))

    def save(self, path: str | Path) -> None:
        if self.network is None:
            raise RuntimeError("Model is not fitted")
        torch = self._torch()
        buffer = io.BytesIO()
        torch.save(self.network.state_dict(), buffer)
        payload = {
            "model_type": "torch_sequence_return",
            "capabilities": self.capabilities.model_dump(mode="json"),
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "state_dict_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TorchSequenceReturnModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        capabilities = ModelCapabilities.model_validate(payload["capabilities"])
        model = cls(
            capabilities.implementation,
            capabilities.feature_names,
            hyperparameters=dict(payload["hyperparameters"]),
            seed=int(payload["seed"]),
            forecast_horizon=capabilities.forecast_horizon,
        )
        torch = model._torch()
        model.network = model._build_network()
        state = torch.load(
            io.BytesIO(base64.b64decode(payload["state_dict_base64"])),
            map_location="cpu",
            weights_only=True,
        )
        model.network.load_state_dict(state)
        return model


def sequence_examples(
    rows: list[FeatureRow],
    context_length: int,
) -> list[tuple[list[list[float]], float]]:
    """Build same-symbol sequences ending at the decision row only."""
    by_symbol: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        by_symbol[row.symbol].append(row)
    examples: list[tuple[list[list[float]], float]] = []
    for symbol_rows in by_symbol.values():
        ordered = sorted(symbol_rows, key=lambda item: item.timestamp)
        for index in range(context_length - 1, len(ordered)):
            current = ordered[index]
            window = ordered[index - context_length + 1 : index + 1]
            if any(item.timestamp > current.timestamp for item in window):
                raise ValueError("Sequence contains future observations")
            examples.append(([list(item.features) for item in window], current.target_return))
    return examples
