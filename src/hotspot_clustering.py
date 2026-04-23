"""
src/hotspot_clustering.py
Compare DBSCAN, HDBSCAN, and KMeans on LA crime coordinates.
Saves cluster plots and centroid CSV.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR


def run_clustering(df: pd.DataFrame, sample_size: int = 60000) -> tuple:
    """
    Run DBSCAN, HDBSCAN, and KMeans. Return results dict, best method name,
    and centroids DataFrame.
    """
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    coords = df[["latitude", "longitude"]].dropna().values

    # Sample for speed on large dataset
    if len(coords) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), sample_size, replace=False)
        coords_s = coords[idx]
    else:
        coords_s = coords

    results = {}

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    print("[Clustering] Running DBSCAN (eps=0.01, min_samples=15)...")
    db = DBSCAN(eps=0.01, min_samples=15, n_jobs=-1).fit(coords_s)
    db_labels = db.labels_
    n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    noise_db = (db_labels == -1).mean() * 100
    mask_db = db_labels != -1
    sil_db = (
        silhouette_score(coords_s[mask_db], db_labels[mask_db], sample_size=5000)
        if n_db > 1 and mask_db.sum() > 10
        else -1.0
    )
    results["dbscan"] = {
        "labels": db_labels, "n_clusters": n_db,
        "silhouette": sil_db, "noise_pct": noise_db,
    }
    print(f"  -> {n_db} clusters | noise={noise_db:.1f}% | silhouette={sil_db:.3f}")

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    print("[Clustering] Running HDBSCAN (min_cluster_size=30)...")
    try:
        import hdbscan as hdbscan_lib
        hdb = hdbscan_lib.HDBSCAN(min_cluster_size=30, min_samples=10, core_dist_n_jobs=-1)
        hdb_labels = hdb.fit_predict(coords_s)
        n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        noise_hdb = (hdb_labels == -1).mean() * 100
        mask_hdb = hdb_labels != -1
        sil_hdb = (
            silhouette_score(coords_s[mask_hdb], hdb_labels[mask_hdb], sample_size=5000)
            if n_hdb > 1 and mask_hdb.sum() > 10
            else -1.0
        )
        results["hdbscan"] = {
            "labels": hdb_labels, "n_clusters": n_hdb,
            "silhouette": sil_hdb, "noise_pct": noise_hdb,
        }
        print(f"  -> {n_hdb} clusters | noise={noise_hdb:.1f}% | silhouette={sil_hdb:.3f}")
    except ImportError:
        print("  HDBSCAN not installed - skipping.")

    # ── KMeans (k=21 matches 21 LAPD divisions) ───────────────────────────────
    print("[Clustering] Running KMeans (k=21, matching LAPD divisions)...")
    km = KMeans(n_clusters=21, random_state=42, n_init=10).fit(coords_s)
    km_labels = km.labels_
    sil_km = silhouette_score(coords_s, km_labels, sample_size=5000)
    results["kmeans"] = {
        "labels": km_labels, "n_clusters": 21,
        "silhouette": sil_km, "noise_pct": 0.0,
        "centers": km.cluster_centers_,
    }
    print(f"  -> 21 clusters | silhouette={sil_km:.3f}")

    # ── Pick best by silhouette score ─────────────────────────────────────────
    scores = {k: v["silhouette"] for k, v in results.items()}
    best_method = max(scores, key=scores.get)
    print(f"\n[Clustering] Best: {best_method.upper()} (silhouette={scores[best_method]:.3f})")

    # ── Compute hotspot centroids from best method ────────────────────────────
    best_labels = results[best_method]["labels"]
    centroids = []
    for lbl in sorted(set(best_labels)):
        if lbl == -1:
            continue
        mask = best_labels == lbl
        centroids.append({
            "cluster":   int(lbl),
            "lat":       float(coords_s[mask, 0].mean()),
            "lon":       float(coords_s[mask, 1].mean()),
            "size":      int(mask.sum()),
            "method":    best_method,
        })
    centroids_df = pd.DataFrame(centroids).sort_values("size", ascending=False)
    centroids_df.to_csv(REPORTS_DIR / "hotspot_centroids.csv", index=False)

    # ── Clustering comparison table ───────────────────────────────────────────
    comparison = pd.DataFrame([
        {
            "method":     k,
            "n_clusters": v["n_clusters"],
            "silhouette": round(v["silhouette"], 4),
            "noise_pct":  round(v.get("noise_pct", 0), 2),
        }
        for k, v in results.items()
    ])
    comparison.to_csv(REPORTS_DIR / "clustering_comparison.csv", index=False)
    print("\nClustering Comparison:")
    print(comparison.to_string(index=False))

    # ── Visualise ─────────────────────────────────────────────────────────────
    _plot_clusters(coords_s, results, best_method)

    return results, best_method, centroids_df


def _plot_clusters(coords: np.ndarray, results: dict, best_method: str):
    valid = [(k, v) for k, v in results.items() if v is not None]
    n_panels = len(valid)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6),
                              facecolor="#0d1117")

    if n_panels == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, valid):
        labels = res["labels"]
        unique_labels = sorted(set(labels))
        cmap = cm.get_cmap("tab20", max(len(unique_labels), 2))

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(coords[mask, 1], coords[mask, 0],
                           c="gray", s=0.2, alpha=0.15, linewidths=0)
            else:
                ax.scatter(coords[mask, 1], coords[mask, 0],
                           c=[cmap(i % 20)], s=0.8, alpha=0.5, linewidths=0)

        is_best = name == best_method
        title = (f"{'★ ' if is_best else ''}{name.upper()}\n"
                 f"{res['n_clusters']} clusters | "
                 f"sil={res['silhouette']:.3f}")
        ax.set_title(title, color="white",
                     fontsize=12, fontweight="bold" if is_best else "normal")
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.set_xlabel("Longitude", color="white")
        ax.set_ylabel("Latitude", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle("LA Crime Hotspot Clustering Comparison",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "hotspot_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Clustering] Cluster plot saved -> {out}")


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "crime_clean.csv")
    results, best, centroids = run_clustering(df)
    print(centroids.head(10))
