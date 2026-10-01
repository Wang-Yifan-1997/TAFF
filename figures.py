"""Static figures for output/results.csv (matplotlib).

    python figures.py      # reads output/results.csv, writes output/figures/*.png
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order

# Plant types in display order (dirty -> clean, bottom -> top of stacks).
TYPE_ORDER = ["coal", "gas", "wind", "solar"]
TYPE_COLOR = dict(zip(TYPE_ORDER, SLOTS))
SCENARIOS = {
    "baseline": ("Baseline (no policy)", INK_2),
    "carbon_tax": ("Carbon tax $30/t", SLOTS[0]),
    "feed_in_premium": ("Feed-in premium $20/MWh, 15 yrs", SLOTS[1]),
    "rd_subsidy": ("Clean R&D subsidy 50%, 15 yrs", SLOTS[2]),
}
HORIZON = 60

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def _end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_2)


def _years(ax):
    ax.set_xlim(0, HORIZON)
    ax.set_xlabel("Years from start")


def baseline_figure(df, path):
    b = df[(df.scenario == "baseline") & (df.year <= HORIZON)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.suptitle("Baseline: the transition with no policy", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)

    ax = axes[0, 0]
    gens = [b[f"generation_{k}"] for k in TYPE_ORDER]
    ax.stackplot(b.year, gens, colors=[TYPE_COLOR[k] for k in TYPE_ORDER],
                 alpha=0.9, edgecolor=SURFACE, linewidth=1.2)
    ax.set_title("Electricity generation by plant type (TWh)")
    _years(ax)

    ax = axes[0, 1]
    for k in TYPE_ORDER:
        ax.plot(b.year, 100 * b[f"build_share_{k}"], color=TYPE_COLOR[k], label=k)
    ax.set_title("Share of developers building each type (%)")
    ax.set_ylim(0, 100)
    _years(ax)

    ax = axes[1, 0]
    for k in TYPE_ORDER:
        ax.plot(b.year, b[f"capex_{k}"], color=TYPE_COLOR[k], label=k)
    ax.set_title("Capital cost of a new plant ($/kW)")
    ax.set_ylim(0, None)
    _years(ax)

    ax = axes[1, 1]
    ax.plot(b.year, 100 * b.innovation_rate_clean, color=INK_2)
    ax.plot(b.year, 100 * b.innovation_rate_dirty, color=MUTED)
    i = int(b.innovation_rate_clean.idxmax() - b.index[0])
    _end_label(ax, b.year.iloc[i], 100 * b.innovation_rate_clean.iloc[i], "clean varieties", dy=6)
    ax.annotate("dirty varieties (research stops)", (7, 0.6), xytext=(6, 10),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_title("Innovation: share of varieties improved each year (%)")
    ax.set_ylim(0, None)
    _years(ax)

    handles = [plt.Line2D([], [], color=TYPE_COLOR[k], lw=6) for k in reversed(TYPE_ORDER)]
    fig.legend(handles, list(reversed(TYPE_ORDER)), loc="upper right", ncol=4,
               fontsize=9, bbox_to_anchor=(0.995, 0.985), handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def policy_figure(df, path):
    d = df[df.year <= HORIZON]
    panels = [
        ("clean_share_of_new_capacity", "Clean share of new capacity (%)", 100),
        ("capex_solar", "Solar capital cost ($/kW)", 1),
        ("emissions_Mt", "CO₂ emissions (Mt per year)", 1),
        ("electricity_price", "Wholesale electricity price ($/MWh)", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.suptitle("Policy scenarios (all announced at year 0)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    for ax, (col, title, scale) in zip(axes.flat, panels):
        for name, (label, color) in SCENARIOS.items():
            s = d[d.scenario == name]
            ax.plot(s.year, scale * s[col], color=color, label=label,
                    zorder=3 if name != "baseline" else 2)
        ax.set_title(title)
        _years(ax)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 1].set_ylim(0, None)
    axes[1, 0].set_ylim(0, None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.005, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(df, outdir):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_figure(df, outdir / "baseline.png")
    policy_figure(df, outdir / "policies.png")
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent / "output"
    make_figures(pd.read_csv(here / "results.csv"), here / "figures")
