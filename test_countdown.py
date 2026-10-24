"""Checks of the answer checker and rewards: python test_countdown.py (or pytest)."""

import random

import countdown as cd

N, T = [3, 7, 25], 46


def outcome(answer, numbers=N, target=T):
    return cd.check(answer, numbers, target)[0]


def test_checker_loopholes():
    assert outcome("<answer>25 + 3 * 7</answer>") == "correct"
    assert outcome("3*7 is 21, plus 25.\n<answer> (3 * 7) + 25 </answer>") == "correct"
    assert (
        outcome("<answer>25 + 3 * 7 + 3</answer>") == "wrong_numbers"
    )  # repeated number
    assert (
        outcome("<answer>25 + 3 * 7</answer>", [3, 7, 25, 1]) == "wrong_numbers"
    )  # skipped number
    assert outcome("<answer>46</answer>") == "wrong_numbers"  # only the target
    assert (
        outcome("<answer>37 + 9</answer>", [3, 7, 9]) == "wrong_numbers"
    )  # joined digits
    assert outcome("<answer>25 + 3 * 7 = 46</answer>") == "invalid"
    assert outcome("<answer>-3 + 7 * 7</answer>", [3, 7, 7]) == "invalid"  # unary minus
    assert outcome("<answer>3 ** 2 + 37</answer>", [3, 2, 37]) == "invalid"
    assert outcome("<answer>2.5 * 2</answer>", [2.5, 2], 5) == "invalid"
    assert outcome("<answer>0x19 + 21</answer>") == "invalid"
    assert outcome("<answer>5 / (3 - 3)</answer>", [3, 3, 5], 1) == "div_zero"
    assert (
        outcome("<answer>1 / 3 * 3</answer>", [1, 3, 3], 1) == "correct"
    )  # exact arithmetic
    assert outcome("25 + 3 * 7") == "no_answer"
    assert (
        outcome("<answer>3 + 7 + 25</answer> no, <answer>25 + 3 * 7</answer>")
        == "correct"
    )  # last tag wins
    assert (
        outcome("<answer>25 + 3 * 7</answer> no, <answer>3 + 7 + 25</answer>") == "near"
    )


def test_rewards():
    def r(kind, answer, numbers=N):
        return cd.reward(kind, *cd.check(answer, numbers, T), numbers, T)

    for kind in cd.REWARDS:
        assert r(kind, "<answer>25 + 3 * 7</answer>") == 1.0
        assert (
            r(kind, "<answer>(25 + 3) * 7</answer>") == 0.0
        )  # 196: closeness clamps at 0
    assert r("binary", "<answer>3 + 7 + 25</answer>") == 0.0
    assert (
        abs(r("partial", "<answer>3 + 7 + 25</answer>") - 0.5 * (1 - 11 / 46)) < 1e-12
    )
    assert r("partial", "<answer>25 + 3 * 7 + 3</answer>") == 0.0
    assert (
        r("lenient", "<answer>25 + 7 + 7 + 7</answer>") == 1.0
    )  # the loophole: repeats pass
    assert (
        r("lenient", "<answer>25 + 3 * 7</answer>", [3, 7, 25, 1]) == 1.0
    )  # skips pass
    assert r("lenient", "<answer>46</answer>") == 0.0  # 46 is not in the set


def test_shuffle_near_keeps_group_multisets():
    prompts = ["a"] * 8 + ["b"] * 8
    outcomes = [
        "near",
        "correct",
        "near",
        "near",
        "invalid",
        "near",
        "no_answer",
        "near",
    ] * 2
    rewards = [
        0.1,
        1.0,
        0.2,
        0.0,
        0.0,
        0.4,
        0.0,
        0.3,
        0.45,
        1.0,
        0.05,
        0.15,
        0.0,
        0.25,
        0.0,
        0.35,
    ]
    out = cd.shuffle_near(rewards, outcomes, prompts, random.Random(0))
    assert sorted(out[:8]) == sorted(rewards[:8]) and sorted(out[8:]) == sorted(
        rewards[8:]
    )
    assert [out[i] for i in (1, 4, 6, 9)] == [1.0, 0.0, 0.0, 1.0]  # non-near untouched
    assert out != rewards
    try:
        cd.shuffle_near(rewards, outcomes, ["a", "b"] * 8, random.Random(0))
    except AssertionError:
        pass
    else:
        raise AssertionError("non-contiguous groups must fail")


def test_generated_tasks():
    sizes = {"x": 300, "y": 300}
    splits = cd.make_splits(sizes, seed=1, n=4, lo=1, hi=25, tmin=10, tmax=100)
    keys = [
        {(tuple(sorted(t["numbers"])), t["target"]) for t in s} for s in splits.values()
    ]
    assert [len(k) for k in keys] == [300, 300] and not keys[0] & keys[1]
    for t in splits["x"] + splits["y"]:
        assert (
            cd.check(f"<answer>{t['solution']}</answer>", t["numbers"], t["target"])[0]
            == "correct"
        )
        assert t["target"] not in t["numbers"] and t["target"] != sum(t["numbers"])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("ok")
