import json


import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path

def plot_overlay_accuracies_per_scenario(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path: Path,
    scenario: str,
    metadata: str,
    ref_clients_acc: float | None = None,
    ref_global_acc: float | None = None,
    font_size: int = 14,
):
    """
    Plot two subplots (clients/global accuracies) per aggregation method,
    each with its own legend inside the plot (bottom-right corner).
    Reference lines are bold for visibility.
    """

    # --- Font settings ---
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # --- Plot clients accuracies ---
    for agg_name, pairs in sorted(results_by_method.items()):
        acc_clients = [a for (a, _) in pairs]
        ax1.plot(acc_clients, linestyle="-", label=f"{agg_name}")

    if ref_clients_acc is not None:
        ax1.axhline(
            ref_clients_acc,
            linestyle=":",
            color="black",
            linewidth=2.5,
            label=f"Ref: clients={ref_clients_acc:.4f}"
        )

    ax1.set_ylabel("Accuracy on Clients Data")
    ax1.set_title("")#f"{scenario.capitalize()}, {metadata.capitalize()}")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Legend inside top plot
    ax1.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.8,
        fancybox=True,
        edgecolor="gray"
    )

    # --- Plot global accuracies ---
    for agg_name, pairs in sorted(results_by_method.items()):
        acc_global = [b for (_, b) in pairs]
        ax2.plot(acc_global, linestyle="--", label=f"{agg_name}")

    if ref_global_acc is not None:
        ax2.axhline(
            ref_global_acc,
            linestyle=":",
            color="black",
            linewidth=2.5,
            label=f"Ref: global={ref_global_acc:.4f}"
        )

    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy on Global Data")
    ax2.set_title("")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Legend inside bottom plot
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.8,
        fancybox=True,
        edgecolor="gray"
    )

    # --- Adjust layout ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"All Aggregation Methods ({scenario.capitalize()}, {metadata.capitalize()})", fontsize=font_size + 2, y=0.995)

    # --- Save high-res plot ---
    output_path = results_path / f"overlay_two_plots_insideleg_boldref_{scenario}_{metadata}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



import json
def extract_from_ctaegory(path: str, keyword: str) -> str:
    idx = path.find(keyword)
    if idx == -1:
        raise ValueError(f"Keyword '{keyword}' not found in path: {path}")
    return path[idx:]

def extract_after_keyword(path: str, keyword: str) -> str:

    idx = path.find(keyword)
    if idx == -1:
        raise ValueError(f"Keyword '{keyword}' not found in path: {path}")
    return path[idx + len(keyword):].lstrip("/\\")

current_dir = Path("")

# get all subfolders (not files)
subfolders = [f for f in current_dir.iterdir() if f.is_dir()]


for subfolder in subfolders:
    with open(f"{subfolder}/summary_0.json", "r", encoding="utf-8") as f:
        json_summary = json.load(f)
    scenario,metadata=str(subfolder).split("_")
    base_line_acc_on_global_data = ""
    base_line_acc_on_clients_data = ""
    results_by_method={}
    for item in json_summary:
        dict_data=json_summary[item]
        base_line_acc_on_global_data=dict_data["global_clients_metric_acc"]
        base_line_acc_on_clients_data=dict_data["global_clients_all_metrics_acc"]
        json_path=extract_from_ctaegory(dict_data["json_path"],f"{scenario}_{metadata}")
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        agg_method=    json_data["agg_method_name"]
        accuracies=json_data["accuracies"]
        results_by_method[agg_method]=accuracies


    plot_overlay_accuracies_per_scenario(results_by_method=results_by_method,results_path=Path(subfolder),scenario=scenario,metadata=metadata,
                                                 ref_global_acc=base_line_acc_on_global_data,
                                                 ref_clients_acc=base_line_acc_on_clients_data
                                                ,font_size=16 )



