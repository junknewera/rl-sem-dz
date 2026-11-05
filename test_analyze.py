"""Check step alignment and the compact replay: python test_analyze.py."""

import json
import tempfile
from pathlib import Path

import pandas as pd

import analyze


with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    run = root / "runs" / "shuffled_s0"
    run.mkdir(parents=True)
    history = [
        {
            "step": s + 1,
            "reward": 0.1,
            "kl": 0.01,
            "completions/mean_length": 10,
            "frac_reward_zero_std": 0.5,
            "completions/clipped_ratio": 0.0,
        }
        for s in range(2)
    ]
    (run / "log_history.json").write_text(json.dumps(history))
    rows = [
        {
            "step": s,
            "outcome": "near" if i < (1 if s == 0 else 8) else "correct",
            "value": "9",
            "target": 10,
            "reward": 0.45,
        }
        for s in range(2)
        for i in range(8)
    ]
    (run / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    frame = analyze.load_training(root / "runs", root / "training.csv")
    assert frame.step.tolist() == [1, 2]
    assert frame.train_acc.tolist() == [7 / 8, 0]
    assert frame.dead.tolist() == [0, 1]
    assert frame.near_count.tolist() == [1, 8]
    assert frame.alone_near_count.tolist() == [1, 0]
    assert frame.unchanged_near_count.tolist() == [1, 8]
    replay = analyze.load_training(root / "absent", root / "training.csv")
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True), replay, check_dtype=False
    )
    saved = (root / "training.csv").read_bytes()
    (run / "log_history.json").write_text(json.dumps(history[:1]))
    try:
        analyze.load_training(root / "runs", root / "training.csv")
    except ValueError as e:
        assert "Raw runs/steps differ" in str(e)
    else:
        raise AssertionError("partial raw data must not overwrite the complete export")
    assert (root / "training.csv").read_bytes() == saved
print("ok")
