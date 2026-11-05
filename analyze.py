"""Figures and tables for the report, from runs/ and results/ (no GPU needed).

python analyze.py    # writes figures/training.png, prints markdown tables
"""

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import countdown as cd

NAMES = {
    "binary": "A binary",
    "shuffled": "C-shuf placebo",
    "partial": "C partial",
    "lenient": "B lenient",
}
RUN = re.compile(r"(\w+)_s(\d+)(?:/checkpoint-(\d+))?")


def parse(model):
    """'runs/partial_s3/checkpoint-25' -> ('partial', 3, 25); 'base' -> ('base', 0, 0)."""
    m = RUN.search(model)
    if model == "base" or not m:
        return "base", 0, 0
    return m[1], int(m[2]), int(m[3] or -1)  # -1: final adapter


def results(path):
    df = pd.DataFrame(cd.load(path)).drop_duplicates(
        ["model", "i"], keep="last"
    )  # evaluate.py appends on reruns
    df[["kind", "seed", "step"]] = df.model.apply(parse).tolist()
    return df


def training(runs):
    """One row per (run, step): logged reward, KL, length, plus strict accuracy and dead groups from completions."""
    rows = []
    for log in Path(runs).glob("*_s*/log_history.json"):
        kind, seed, _ = parse(log.parent.name)
        steps = pd.DataFrame([h for h in json.loads(log.read_text()) if "kl" in h])
        steps = steps[
            [
                "step",
                "reward",
                "kl",
                "completions/mean_length",
                "frac_reward_zero_std",
                "completions/clipped_ratio",
            ]
        ].rename(
            columns={
                "completions/mean_length": "length",
                "frac_reward_zero_std": "zero_std",
                "completions/clipped_ratio": "clipped",
            }
        )
        done = pd.DataFrame(cd.load(log.parent / "completions.jsonl"))
        done["correct"] = done.outcome == "correct"
        done["group"] = (
            done.index // cd.G
        )  # the G completions of one prompt are next to each other
        per_step = done.groupby("step").correct.mean().rename("train_acc").to_frame()
        solved = done.groupby(
            ["step", "group"]
        ).correct.any()  # did any of the G answers solve the task
        per_step["dead"] = 1 - solved.groupby("step").mean()
        near = done.outcome == "near"
        alone = near.groupby(done.group).transform("sum") <= 1
        per_step["near_count"] = near.groupby(done.step).sum()
        per_step["alone_near_count"] = (near & alone).groupby(done.step).sum()
        if kind == "shuffled":
            original = pd.Series(
                [
                    0.5 * cd.closeness(Fraction(v), t)
                    for v, t in done.loc[near, ["value", "target"]].itertuples(
                        index=False, name=None
                    )
                ],
                index=done.index[near],
            )
            unchanged = (done.loc[near, "reward"] - original).abs() < 1e-12
            tied = done.loc[near].groupby("group").reward.transform("nunique") == 1
            per_step["unchanged_near_count"] = unchanged.groupby(
                done.loc[near, "step"]
            ).sum()
            per_step["tied_near_count"] = tied.groupby(done.loc[near, "step"]).sum()
        per_step.index += 1  # rewards at global_step s train optimizer step s + 1
        rows.append(steps.join(per_step, on="step").assign(kind=kind, seed=seed))
    return pd.concat(rows)


def load_training(runs, path):
    if list(Path(runs).glob("*_s*/log_history.json")):
        frame = training(runs)
        if Path(path).exists():
            keys = ["kind", "seed", "step"]
            saved = pd.read_csv(path, usecols=keys)
            if set(frame[keys].itertuples(index=False, name=None)) != set(
                saved[keys].itertuples(index=False, name=None)
            ):
                raise ValueError(
                    "Raw runs/steps differ from training.csv; replay with --runs pointing to an absent directory, "
                    "or use a separate --results directory for a new experiment"
                )
        frame.to_csv(path, index=False)
        return frame
    return pd.read_csv(
        path
    )  # published compact metrics reproduce figures without raw training completions


def band(ax, df, col, title, ylabel, smooth=1):
    """Mean over seeds with a +-1 std band, each seed smoothed by a rolling mean over `smooth` steps."""
    for kind, name in NAMES.items():
        if kind not in set(df.kind):
            continue
        per_seed = df[df.kind == kind].pivot_table(
            index="step", columns="seed", values=col
        )
        per_seed = per_seed.rolling(smooth, min_periods=1).mean()
        mean, std = per_seed.mean(axis=1), per_seed.std(axis=1).fillna(0)
        ax.plot(mean.index, mean, label=f"{name} (n={per_seed.shape[1]} seeds)")
        lower, upper = (mean - std).clip(lower=0), mean + std
        if col in ("reward", "train_acc", "acc", "dead"):
            upper = upper.clip(upper=1)
        ax.fill_between(mean.index, lower, upper, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def answer(text):
    found = cd.ANSWER.findall(text)
    return found[-1].strip().replace("|", "/") if found else "-"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--smooth", type=int, default=10)
    args = p.parse_args()
    res = Path(args.results)

    train = load_training(args.runs, res / "training.csv")
    val = results(res / "val.jsonl")
    val = (
        val[val.step >= 0]
        .groupby(["kind", "seed", "step"])
        .outcome.apply(lambda o: (o == "correct").mean())
    )
    val = val.rename("acc").reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    band(
        axes[0, 0],
        train,
        "reward",
        "Training reward (each condition's own reward)",
        "mean reward",
        args.smooth,
    )
    band(
        axes[0, 1],
        train,
        "train_acc",
        "Strict accuracy on training samples",
        "share correct",
        args.smooth,
    )
    band(
        axes[0, 2],
        val[val.kind != "base"],
        "acc",
        "Strict accuracy on held-out val (200 tasks, greedy)",
        "share correct",
    )
    axes[0, 2].axhline(
        val[val.kind == "base"].acc.mean(),
        color="gray",
        linestyle="--",
        label="base model",
    )
    band(
        axes[1, 0],
        train,
        "kl",
        "KL to the initial model",
        "KL, nats per token",
        args.smooth,
    )
    axes[1, 0].set_yscale("symlog", linthresh=0.1)
    axes[1, 0].set_title("Sampled KL to initial model (symlog, linear below 0.1)")
    band(axes[1, 1], train, "length", "Mean completion length", "tokens", args.smooth)
    band(
        axes[1, 2],
        train,
        "dead",
        f"Groups with 0 of {cd.G} answers strictly correct",
        "share of groups",
        args.smooth,
    )
    fig.legend(*axes[0, 2].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.suptitle(
        f"Mean +-1 std over seeds (bands clipped to valid range); training smoothing: {args.smooth} steps"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/training.png", dpi=120)

    test = results(res / "test.jsonl")
    test = test[
        (test.step == -1) | (test.kind == "base")
    ]  # final adapters and the base model
    shares = pd.crosstab([test.kind, test.seed], test.outcome, normalize="index")
    acc = shares["correct"]
    shares.reset_index().to_csv(res / "test_by_seed.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, kind in enumerate(["base", *NAMES]):
        values = acc.loc[kind] * 100
        positions = [i + (j - (len(values) - 1) / 2) * 0.05 for j in range(len(values))]
        ax.scatter(
            positions,
            values,
            color="gray" if kind == "base" else f"C{i - 1}",
            alpha=0.8,
        )
        ax.errorbar(
            i,
            values.mean(),
            yerr=values.std() if len(values) > 1 else 0,
            fmt="_",
            color="black",
            capsize=5,
        )
    ax.set_xticks(
        range(5),
        ["Base (1)", "A binary (5)", "C-shuf (8)", "C partial (8)", "B lenient (1)"],
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Strict test accuracy, %")
    ax.set_title(
        "2000 fixed test tasks, greedy; dots = seeds, black bars = mean +-1 std"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/test.png", dpi=120)
    summary = shares.groupby("kind").mean().round(3)
    summary.insert(0, "n_seeds", acc.groupby("kind").size())
    summary.insert(1, "acc_std", acc.groupby("kind").std().round(3))
    print("## Test accuracy (2000 tasks, greedy), outcome shares averaged over seeds\n")
    print(summary.rename(index=NAMES).to_markdown(), "\n")

    print("## Differences in test accuracy, 95% Welch CI over seeds\n")
    for a, b in [
        ("partial", "shuffled"),
        ("partial", "binary"),
        ("shuffled", "binary"),
    ]:
        x, y = acc.get(a), acc.get(b)
        if x is None or y is None or len(x) < 2 or len(y) < 2:
            continue
        lo, hi = stats.ttest_ind(x, y, equal_var=False).confidence_interval(0.95)
        diff = x.mean() - y.mean()
        if -0.03 < lo and hi < 0.03:  # decision rule from PLAN.md, in this order
            verdict = "no practical effect"
        elif lo > 0:
            verdict = "positive"
        elif hi < 0:
            verdict = "negative"
        else:
            verdict = "undetermined"
        print(
            f"- {NAMES[a]} - {NAMES[b]}: {100 * diff:+.1f} pp, CI [{100 * lo:+.1f}, {100 * hi:+.1f}] -> {verdict}"
        )

    paired = acc.unstack("kind")[["partial", "shuffled"]].dropna()
    ci = stats.ttest_rel(paired.partial, paired.shuffled).confidence_interval(0.95)
    print(
        f"\nPost-hoc paired-seed difference, C - C-shuf: CI [{100 * ci.low:+.2f}, {100 * ci.high:+.2f}] pp"
    )

    placebo = train[train.kind == "shuffled"]
    total_near = placebo.near_count.sum()
    if total_near:
        print(
            f"\nC-shuf, among near answers: singletons {placebo.alone_near_count.sum() / total_near:.1%}; "
            f"identical rewards in group {placebo.tied_near_count.sum() / total_near:.1%}; "
            f"assigned reward unchanged {placebo.unchanged_near_count.sum() / total_near:.1%}"
        )

    print("\n## Same test tasks before and after training (seed 0, final adapters)\n")
    rows = test[(test.i < 10) & (test.seed == 0)]
    rows = rows.assign(cell=rows.completion.apply(answer) + " (" + rows.outcome + ")")
    table = rows.pivot_table(index="i", columns="kind", values="cell", aggfunc="first")
    tasks = rows.drop_duplicates("i").set_index("i")
    table.insert(
        0, "task", tasks.numbers.astype(str) + " -> " + tasks.target.astype(str)
    )
    print(
        table[["task"] + [k for k in ["base", *NAMES] if k in table]]
        .rename(columns=NAMES)
        .to_markdown(),
        "\n",
    )

    print("## Error examples (C partial, seed 0, test)\n")
    c0 = test[(test.kind == "partial") & (test.seed == 0)]
    for outcome, g in c0[c0.outcome != "correct"].groupby("outcome"):
        r = g.iloc[0]
        print(f"- {outcome}: {r.numbers} -> {r.target}: `{r.completion[:200]!r}`")

    b = test[test.kind == "lenient"]
    if len(b):
        lenient = []
        for r in b.itertuples():
            outcome, value, used = cd.check(r.completion, r.numbers, r.target)
            lenient.append(
                cd.reward("lenient", outcome, value, used, r.numbers, r.target)
            )
        hacked = b[(pd.Series(lenient, index=b.index) == 1) & (b.outcome != "correct")]
        print(
            f"\n## Reward hacking in B: {len(hacked)} of {len(b)} test answers get reward 1 but fail the strict check\n"
        )
        for r in hacked.head(5).itertuples():
            print(f"- {r.numbers} -> {r.target}: `{answer(r.completion)}`")
