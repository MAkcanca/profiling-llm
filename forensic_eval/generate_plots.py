import os
from pathlib import Path
import json

from forensic_eval.reporting.visualization_new import create_visualizations


def main():
    # Gets the latest folder created inside paper_results\scientific_analysis folder
    scientific_dir = Path("paper_results/scientific_analysis")
    if not scientific_dir.exists():
        print(f"Directory {scientific_dir} does not exist")
        return
    # Get all folders in the directory
    folders = [f for f in scientific_dir.iterdir() if f.is_dir()] 
    # Sort folders by creation time and get the latest
    latest_folder = max(folders, key=lambda x: x.stat().st_birthtime)
    print(f"Using latest folder: {latest_folder}")
    plots_dir = latest_folder / "plots"
    results = json.load(open(latest_folder / "evaluation_results.json"))
    create_visualizations(results, plots_dir)

if __name__ == "__main__":
    main()