import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_summary_table(json_path: Path, figsize=(8, 3), font_size=20, dpi=200):
    """
    Plot the __overall_means__ section from outlier_summary.json as a table.
    The first column (metric names) has a darker gray background,
    white bold text, and a slightly larger font than the numeric column.
    """

    # --- Load data ---
    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("__overall_means__", None)
    if summary is None:
        raise KeyError("__overall_means__ not found in JSON file")

    # --- Prepare table data ---
    table_data = [
        ["Mean Accuracy on Global Data", f"{summary['mean_global_test_accuracy']:.3f}"],
        ["Mean Accuracy on Clients Data", f"{summary['mean_all_client_test_accuracies']:.3f}"],
        ["Mean Max Eval Accuracy", f"{summary['mean_best_eval_accuracy']:.3f}"],
    ]

    # --- Plot table ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
    )

    # --- Basic styling ---
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # --- Adjust individual cell sizes and styles ---
    for (row, col), cell in table.get_celld().items():
        width, height = cell.get_width(), cell.get_height()
        if col == 0:  # first column
            cell.set_width(width * 2.35)
            cell.set_facecolor("#555555")  # darker gray
            text_obj = cell.get_text()
            text_obj.set_color("white")
            text_obj.set_fontsize(font_size * 1.2)
            text_obj.set_fontweight("bold")
        else:  # numeric column
            cell.set_width(width * 0.5)
            cell.get_text().set_fontsize(font_size)
            cell.get_text().set_fontweight("bold")

        cell.set_height(height * 1.7)

    plt.tight_layout()
    plt.savefig("summary_table_plot.png", dpi=dpi)
    plt.show()


# === Usage ===
plot_summary_table(Path("outlier_summary.json"))
