import os
import sys
import time
import re
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTHOR_LOGIN = "Kizorat"

REPOS = [
    "Kizorat/CulturalMonument-TBox",
    "TheOverfitters/ContextWare",
    "Kizorat/DeepFake",
    "Kizorat/LearningGrid",
    "Kizorat/HeliPathGIS",
    "Kizorat/KeybladeDB",
]

EXT_TO_LANG = {
    ".py": "Python",
    ".ipynb": "Jupyter Notebook",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".md": "Markdown",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".sh": "Shell",
    ".sql": "SQL",
}

LANG_COLORS = {
    "Python": "#378ADD",
    "Jupyter Notebook": "#BA7517",
    "JavaScript": "#EF9F27",
    "TypeScript": "#3B6D11",
    "HTML": "#D85A30",
    "CSS": "#7F77DD",
    "JSON": "#888780",
    "Markdown": "#5DCAA5",
    "YAML": "#D4537E",
    "Shell": "#E24B4A",
    "SQL": "#FAC775",
}


PACKAGE_TO_TOOL = {
    "torch": "PyTorch",
    "torchvision": "PyTorch",
    "torchaudio": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf-keras": "TensorFlow",
    "keras": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "transformers": "Hugging Face",
    "trl": "Hugging Face",
    "peft": "Hugging Face",
    "datasets": "Hugging Face",
    "huggingface-hub": "Hugging Face",
    "llama-cpp-python": "llama.cpp",
    "ollama": "Ollama",
    "openai": "OpenAI SDK",
    "gymnasium": "Gymnasium (RL)",
    "minigrid": "Gymnasium (RL)",
    "fastapi": "FastAPI",
    "uvicorn": "FastAPI",
    "flask": "Flask",
    "pymongo": "MongoDB",
    "owlready2": "OWL/Semantic Web",
    "owlrl": "OWL/Semantic Web",
    "rdflib": "OWL/Semantic Web",
    "geopandas": "GeoPandas/GIS",
    "shapely": "GeoPandas/GIS",
    "pyproj": "GeoPandas/GIS",
    "pyogrio": "GeoPandas/GIS",
    "networkx": "NetworkX",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Matplotlib",
}

TOOL_COLORS = {
    "PyTorch": "#D85A30",
    "TensorFlow": "#EF9F27",
    "scikit-learn": "#378ADD",
    "Hugging Face": "#FAC775",
    "llama.cpp": "#888780",
    "Ollama": "#2C2C2A",
    "OpenAI SDK": "#5DCAA5",
    "Gymnasium (RL)": "#639922",
    "FastAPI": "#04342C",
    "Flask": "#7F77DD",
    "MongoDB": "#27500A",
    "OWL/Semantic Web": "#D4537E",
    "GeoPandas/GIS": "#0C447C",
    "NetworkX": "#A32D2D",
    "Pandas": "#3C3489",
    "NumPy": "#085041",
    "Matplotlib": "#993C1D",
}

BG = "#0d1117"
FG = "#c9d1d9"

API = "https://api.github.com"


def gh_headers(token):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_commit_shas(repo, author, token):
    """List all commit SHAs by `author` in `repo`, paginated."""
    shas = []
    page = 1
    while True:
        url = f"{API}/repos/{repo}/commits"
        params = {"author": author, "per_page": 100, "page": page}
        resp = requests.get(url, headers=gh_headers(token), params=params, timeout=20)
        if resp.status_code == 409:
            # empty repository
            break
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        shas.extend(c["sha"] for c in batch)
        if len(batch) < 100:
            break
        page += 1
    return shas


def get_commit_file_stats(repo, sha, token):
    """Return list of (filename, additions) for a single commit."""
    url = f"{API}/repos/{repo}/commits/{sha}"
    resp = requests.get(url, headers=gh_headers(token), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("files", [])
    return [(f["filename"], f.get("additions", 0)) for f in files]


def get_requirements_packages(repo, token):
    """Fetch requirements.txt from the default branch and return a list of
    lowercase package names (best-effort parsing, ignores version pins,
    extras, and local/file-path requirements)."""
    url = f"{API}/repos/{repo}/contents/requirements.txt"
    resp = requests.get(url, headers=gh_headers(token), timeout=20)
    if resp.status_code != 200:
        return []
    data = resp.json()
    download_url = data.get("download_url")
    if not download_url:
        return []
    raw = requests.get(download_url, timeout=20).text

    packages = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "@" in line or "file:///" in line:
            # local/conda build artifact lines, e.g. "pkg @ file:///..."
            name = line.split("@")[0].strip()
        else:
            name = re.split(r"[=<>!~\[; ]", line)[0].strip()
        if name:
            packages.append(name.lower())
    return packages


def lang_for_filename(filename):
    for ext, lang in EXT_TO_LANG.items():
        if filename.endswith(ext):
            return lang
    return None


def make_pie(sizes, labels, colors, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(BG)

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": BG},
    )
    legend_labels = [f"{l} {s:.1f}%" for l, s in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=11,
        frameon=False,
        labelcolor=FG,
    )
    ax.set_title(title, color=FG, fontsize=13, pad=14)

    plt.tight_layout()
    plt.savefig(out_path, format="svg", facecolor=BG, bbox_inches="tight")
    print(f"Saved {out_path}")


def make_bar_chart(counts, labels, colors, title, out_path):
    # sort descending so the most-used tool appears at the top
    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    counts = [counts[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]

    height = max(3.5, 0.5 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, height), facecolor=BG)
    ax.set_facecolor(BG)

    y_pos = range(len(labels))
    ax.barh(y_pos, counts, color=colors, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color=FG, fontsize=11)
    ax.invert_yaxis()  # highest count on top

    ax.set_xlabel("Repos using this tool", color=FG, fontsize=10)
    ax.tick_params(axis="x", colors=FG, labelsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#30363d", linewidth=0.5)
    ax.set_axisbelow(True)

    for i, count in enumerate(counts):
        ax.text(count + max(counts) * 0.02, i, str(count), color=FG, va="center", fontsize=9)

    ax.set_title(title, color=FG, fontsize=13, pad=14)

    plt.tight_layout()
    plt.savefig(out_path, format="svg", facecolor=BG, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    token = os.environ.get("GITHUB_TOKEN")
    lang_totals = {}
    tool_totals = {}
    repos_with_my_commits = []

    for repo in REPOS:
        print(f"Processing {repo}...", file=sys.stderr)
        try:
            shas = get_commit_shas(repo, AUTHOR_LOGIN, token)
        except Exception as exc:
            print(f"  Failed to list commits: {exc}", file=sys.stderr)
            continue

        print(f"  {len(shas)} commits by {AUTHOR_LOGIN}", file=sys.stderr)
        if not shas:
            continue
        repos_with_my_commits.append(repo)

        for sha in shas:
            try:
                file_stats = get_commit_file_stats(repo, sha, token)
            except Exception as exc:
                print(f"  Failed to fetch commit {sha}: {exc}", file=sys.stderr)
                continue

            for filename, additions in file_stats:
                lang = lang_for_filename(filename)
                if lang is None:
                    continue
                lang_totals[lang] = lang_totals.get(lang, 0) + additions

            time.sleep(0.05)  # be gentle with rate limits

    # Tools/frameworks: only from repos where the author has at least one commit
    for repo in repos_with_my_commits:
        try:
            packages = get_requirements_packages(repo, token)
        except Exception as exc:
            print(f"  Failed to fetch requirements.txt for {repo}: {exc}", file=sys.stderr)
            continue
        seen_in_repo = set()
        for pkg in packages:
            tool = PACKAGE_TO_TOOL.get(pkg)
            if tool is None:
                continue
            if tool in seen_in_repo:
                continue
            seen_in_repo.add(tool)
            tool_totals[tool] = tool_totals.get(tool, 0) + 1

    out_dir = "profile-summary-card-output"
    os.makedirs(out_dir, exist_ok=True)

    if lang_totals:
        total = sum(lang_totals.values())
        labels = list(lang_totals.keys())
        sizes = [v / total * 100 for v in lang_totals.values()]
        colors = [LANG_COLORS.get(l, "#888780") for l in labels]
        make_pie(sizes, labels, colors, "Language Usage",
                 os.path.join(out_dir, "my-language-pie.svg"))
    else:
        print("No language data collected; skipping language chart.", file=sys.stderr)

    if tool_totals:
        labels = list(tool_totals.keys())
        counts = list(tool_totals.values())
        colors = [TOOL_COLORS.get(l, "#888780") for l in labels]
        make_bar_chart(counts, labels, colors, "Tools & Frameworks Usage",
                        os.path.join(out_dir, "my-tools-pie.svg"))
    else:
        print("No tool/framework data collected; skipping tools chart.", file=sys.stderr)


if __name__ == "__main__":
    main()