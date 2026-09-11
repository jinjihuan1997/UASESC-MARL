"""Create diagnostic scientific figures from the saved pilot analysis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
result = json.loads((HERE / "results/results.json").read_text())
curves = json.loads((HERE / "results/training_curves.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 180})
colors = {"R": "#7a7a7a", "G": "#c67816", "A": "#296ea3", "B": "#147d64"}
labels = {"R": "R\nFixed rule", "G": "G\nGreedy rule", "A": "A\nNo explicit input",
          "B": "B\nExplicit input"}

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
groups = [r for r in result["summaries"] if r["scenario"] == "all_scenarios"]
for i, row in enumerate(groups):
    group = row["group"]
    axes[0].bar(i, row["mean"]["reward"], color=colors[group], alpha=.8, width=.65)
    values = [s["reward"] for s in row["per_training_seed"]]
    offsets = np.linspace(-.15, .15, len(values)) if len(values) > 1 else [0]
    axes[0].scatter(i+np.asarray(offsets), values, color="black", s=23, zorder=4)
    axes[0].text(i, max(values)+.008, f'{row["mean"]["reward"]:.3f}', ha="center", fontsize=10)
axes[0].set_xticks(range(4), [labels[g] for g in [r["group"] for r in groups]])
axes[0].set_ylabel("Mean reward per slot (higher is better)")
axes[0].axhline(0, color="#555555", lw=.7)
axes[0].set_ylim(-.045, .28)
axes[0].set_title("Five scenarios, 20 common evaluation seeds")
axes[0].grid(axis="y", alpha=.15)
contrast_names = {"B-A": "With instruction − without", "A-G": "Without instruction − greedy", "B-G": "With instruction − greedy"}
for i, name in enumerate(contrast_names):
    row = next(r for r in result["contrasts"] if r["scenario"] == "all_scenarios" and r["contrast"] == name)
    effect = row["effect"]["reward"]
    lo, hi = row["training_seed_t_95_ci_conditional_on_eval_set"]["reward"]
    axes[1].errorbar(effect, i, xerr=[[effect-lo], [hi-effect]], fmt="D", color="#222222", capsize=4, ms=5)
    axes[1].scatter([r["reward"] for r in row["per_training_seed_effect"]], [i+.13]*3,
                    s=25, color=colors["B"] if name.startswith("B") else colors["A"], zorder=3)
axes[1].axvline(0, color="#777777", lw=1, ls="--")
axes[1].set_yticks(range(3), list(contrast_names.values()))
axes[1].invert_yaxis()
axes[1].set_xlabel("Paired reward difference")
axes[1].set_title("Mean and 95% t interval\nColored dots: paired training-seed effects (n = 3)", fontsize=11)
axes[1].xaxis.set_major_locator(MaxNLocator(5))
axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes[1].grid(axis="x", alpha=.15)
fig.suptitle("Pilot results: explicit instruction gains are not established", fontsize=13)
save(fig, "pilot_comparison")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True, sharey=True, constrained_layout=True)
seed_colors = ["#296ea3", "#c67816", "#147d64"]
for j, group in enumerate(["A", "B"]):
    for policy, row in curves.items():
        if not policy.startswith(group+"_"):
            continue
        seed = int(policy[-1])
        x, y = np.asarray(row["steps"])/1e6, np.asarray(row["reward"])
        axes[j].plot(x, y, alpha=.18, lw=.8, color=seed_colors[seed-1])
        axes[j].plot(x[4:], np.convolve(y, np.ones(5)/5, mode="valid"),
                     color=seed_colors[seed-1], lw=1.6, label=f"Seed {seed}")
    axes[j].set_title("A: no explicit instruction" if group == "A" else "B: explicit instruction")
    axes[j].set_xlabel("Training environment steps (millions)")
    axes[j].grid(alpha=.15)
    axes[j].legend(frameon=False, fontsize=9)
axes[0].set_ylabel("On-policy training reward per slot")
fig.suptitle("Training improves reward; convergence is not established by this pilot", fontsize=12)
save(fig, "training_curves")

trajectories = json.loads((HERE / "results/mean_trajectories.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), sharex=True, constrained_layout=True)
for col, scenario in enumerate(["switch300_aoi", "switch300_quality"]):
    for group in ["R", "G", "A", "B"]:
        items = [np.asarray(r["values"]) for r in trajectories
                 if r["scenario"] == scenario and (r["policy"].startswith(group+"_") or
                    (group in ["R", "G"] and r["policy"].startswith(group+"_")))]
        values = np.mean(items, axis=0)
        x = np.arange(600)
        axes[0, col].plot(x[9:], np.convolve(values[1], np.ones(10)/10, mode="valid"), color=colors[group], label=group)
        delivered = np.convolve(values[2], np.ones(10), mode="valid")
        quality = np.convolve(values[3], np.ones(10), mode="valid") / delivered
        axes[1, col].plot(x[9:], quality, color=colors[group], label=group)
    axes[0, col].set_title("Balance → AoI" if scenario.endswith("aoi") else "Balance → quality")
    for row in range(2):
        axes[row, col].axvline(300, color="#777777", lw=.8, ls="--")
        axes[row, col].grid(alpha=.15)
    axes[1, col].set_xlabel("Slot (instruction changes at 300)")
axes[0, 0].set_ylabel("Mean AoI (slots; lower is better)")
axes[1, 0].set_ylabel("Predicted delivered PSNR (dB)")
axes[0, 1].legend(title="R fixed / G greedy / A no input / B input", frameon=False, ncol=4, fontsize=9, title_fontsize=8)
fig.suptitle("Instruction switch: averaged traces, 10-slot smoothing", fontsize=12)
save(fig, "instruction_switch")
print(f"Saved three figure pairs to {OUT}")
