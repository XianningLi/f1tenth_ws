"""PyTorch checkpoint adapter used by the RPC ROS node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import FEATURE_SIZE, VehicleLimits


EXPECTED_FEATURE_LAYOUT = [
    "x_relative",
    "y_error",
    "speed",
    "yaw",
    "y_reference_relative",
    "speed_reference",
]


class NeuralController:
    """Load an RPC/HFRPC state dictionary and perform one-step inference."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required by rpc_node. Install a PyTorch build "
                "compatible with this computer before launching the node."
            ) from exc

        self._torch = torch
        requested_device = str(device)
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        checkpoint_file = Path(checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"RPC checkpoint not found: {checkpoint_file}")

        checkpoint = self._load_checkpoint(checkpoint_file)
        self.method = str(checkpoint.get("method", ""))
        if self.method not in {"rpc", "hfrpc"}:
            raise ValueError(
                f"unsupported checkpoint method {self.method!r}; expected rpc or hfrpc"
            )

        feature_layout = checkpoint.get("feature_layout", EXPECTED_FEATURE_LAYOUT)
        if list(feature_layout) != EXPECTED_FEATURE_LAYOUT:
            raise ValueError(f"unsupported checkpoint feature layout: {feature_layout}")

        vehicle_config = checkpoint["vehicle"]
        self.vehicle = VehicleLimits(
            wheelbase=float(vehicle_config["wheelbase"]),
            min_speed=float(vehicle_config["min_speed"]),
            max_speed=float(vehicle_config["max_speed"]),
            max_accel=float(vehicle_config["max_accel"]),
            max_steer=float(vehicle_config["max_steer"]),
        )

        mpc_config = checkpoint["mpc"]
        self.horizon = int(mpc_config["horizon"])
        self.model_dt = float(mpc_config["model_dt"])
        if self.horizon < 1 or self.model_dt <= 0.0:
            raise ValueError("invalid horizon or model_dt in checkpoint")

        hidden_size = int(checkpoint.get("training", {}).get("hidden_size", 256))
        self.model = self._build_model(self.method, vehicle_config, hidden_size)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        try:
            return self._torch.load(
                str(path), map_location=self.device, weights_only=True
            )
        except TypeError:
            # PyTorch versions before the weights_only argument was introduced.
            return self._torch.load(str(path), map_location=self.device)

    def _build_model(
        self,
        method: str,
        vehicle_config: dict[str, Any],
        hidden_size: int,
    ) -> Any:
        torch = self._torch
        nn = torch.nn
        max_accel = float(vehicle_config["max_accel"])
        max_steer = float(vehicle_config["max_steer"])

        class MLP(nn.Module):
            def __init__(self, output_size: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(FEATURE_SIZE, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, features: Any) -> Any:
                return self.layers(features)

        def scale_controls(raw_controls: Any) -> Any:
            bounded = torch.tanh(raw_controls)
            acceleration = bounded[..., 0] * max_accel
            steering = bounded[..., 1] * max_steer
            return torch.stack((acceleration, steering), dim=-1)

        class OneStepPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = MLP(2)

            def forward(self, features: Any) -> Any:
                return scale_controls(self.network(features))

        class FeedbackPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = MLP(8)
                self.softplus = nn.Softplus()

            def forward(self, features: Any) -> Any:
                state = features[..., :4]
                reference = features[..., 4:]
                raw_gain = self.network(features).reshape(-1, 2, 4)
                zeros = torch.zeros_like(raw_gain[:, 0, 0])
                acceleration_gain = torch.stack(
                    (
                        zeros,
                        zeros,
                        -self.softplus(raw_gain[:, 0, 2]) - 0.1,
                        raw_gain[:, 0, 3],
                    ),
                    dim=-1,
                )
                steering_gain = torch.stack(
                    (
                        zeros,
                        -self.softplus(raw_gain[:, 1, 1]),
                        zeros,
                        raw_gain[:, 1, 3],
                    ),
                    dim=-1,
                )
                gain = torch.stack((acceleration_gain, steering_gain), dim=1)
                state_error = torch.stack(
                    (
                        torch.zeros_like(state[:, 0]),
                        state[:, 1] - reference[:, 0],
                        state[:, 2] - reference[:, 1],
                        state[:, 3],
                    ),
                    dim=-1,
                )
                raw_control = torch.bmm(
                    gain, state_error.unsqueeze(-1)
                ).squeeze(-1)
                return scale_controls(raw_control)

        return FeedbackPolicy() if method == "hfrpc" else OneStepPolicy()

    def act(self, features: np.ndarray) -> np.ndarray:
        """Return one bounded [acceleration, steering_angle] command."""

        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.shape != (FEATURE_SIZE,):
            raise ValueError(f"features must have shape ({FEATURE_SIZE},)")
        tensor = self._torch.from_numpy(feature_array).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            output = self.model(tensor)
        return output.squeeze(0).detach().cpu().numpy().astype(float, copy=False)
