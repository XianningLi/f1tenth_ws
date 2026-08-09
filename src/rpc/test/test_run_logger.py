import csv
import re

import numpy as np

from rpc.run_logger import RunCsvLogger


def test_run_logger_uses_english_timestamped_name_and_flushed_rows(tmp_path):
    logger = RunCsvLogger(tmp_path, "RPC", horizon=2)
    reference = np.arange(12, dtype=float).reshape(4, 3)
    prediction = reference + 1.0
    controls = np.arange(4, dtype=float).reshape(2, 2)

    logger.log(
        ros_time_sec=12.5,
        status="ok",
        timings={
            "controller_compute_ms": 0.8,
            "solver_internal_ms": None,
            "control_inference_ms": 0.2,
            "prediction_rollout_ms": 0.6,
            "callback_total_ms": 1.0,
        },
        state=np.array([1.0, 2.0, 3.0, 4.0]),
        reference=reference,
        predicted_states=prediction,
        predicted_controls=controls,
        command_acceleration=0.1,
        command_steering=0.2,
        command_speed=3.1,
    )

    assert re.fullmatch(r"RPC_\d{8}_\d{6}_\d{6}\.csv", logger.path.name)
    with logger.path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["control_inference_ms"] == "0.2"
    assert rows[0]["solver_internal_ms"] == ""
    assert rows[0]["predicted_2_yaw"] == str(prediction[3, 2])
    assert not re.search(r"[\u4e00-\u9fff]", logger.path.read_text(encoding="utf-8"))
    logger.close()

