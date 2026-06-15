import csv
import json
from pathlib import Path

out_dir = Path(r"E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss")
history_path = out_dir / "loss_history_160m_5b.json"
csv_path = out_dir / "loss_history_160m_5b.csv"
png_path = out_dir / "loss_curve_160m_5b.png"
svg_path = out_dir / "loss_curve_160m_5b.svg"
history = json.loads(history_path.read_text(encoding="utf-8-sig"))
history = sorted(history, key=lambda row: float(row["tokens_seen"]))
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tokens_b", "step", "train_loss", "val_loss", "learning_rate"])
    for row in history:
        writer.writerow([
            float(row["tokens_seen"]) / 1e9,
            int(float(row["step"])),
            float(row["train_loss"]),
            float(row["val_loss"]),
            float(row.get("learning_rate", 0.0)),
        ])
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [float(r["tokens_seen"]) / 1e9 for r in history]
    train = [float(r["train_loss"]) for r in history]
    val = [float(r["val_loss"]) for r in history]
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    ax.plot(xs, val, color="#0f766e", linewidth=2.4, marker="o", markersize=3.5, label="validation loss")
    ax.plot(xs, train, color="#7c3aed", linewidth=1.5, alpha=0.55, marker=".", markersize=3, label="train loss at eval")
    ax.axvline(2.0, color="#334155", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(2.02, max(val) - 0.05, "fresh-data continuation starts", rotation=90, va="top", ha="left", fontsize=9, color="#334155")
    ax.scatter([xs[-1]], [val[-1]], s=58, color="#dc2626", zorder=5, label=f"final val {val[-1]:.4f}")
    ax.set_title("160M LongSeq Anchor Training Loss to 5B Tokens", fontsize=14, pad=14)
    ax.set_xlabel("Total tokens seen (billions)")
    ax.set_ylabel("Loss")
    ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
    ymin = min(val + train) - 0.08
    ymax = max(val + train) + 0.08
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    print(f"PLOT_OK png={png_path} csv={csv_path} points={len(history)} final_val={val[-1]:.4f}")
except Exception as exc:
    xs = [float(r["tokens_seen"]) / 1e9 for r in history]
    val = [float(r["val_loss"]) for r in history]
    train = [float(r["train_loss"]) for r in history]
    width, height = 1200, 700
    pad_l, pad_r, pad_t, pad_b = 80, 30, 50, 70
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(val + train), max(val + train)
    min_y -= 0.08; max_y += 0.08
    def sx(x): return pad_l + (x - min_x) / (max_x - min_x) * (width - pad_l - pad_r)
    def sy(y): return height - pad_b - (y - min_y) / (max_y - min_y) * (height - pad_t - pad_b)
    def points(vals): return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, vals))
    boundary_x = sx(2.0)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">160M LongSeq Anchor Training Loss to 5B Tokens</text>
<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#334155"/>
<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#334155"/>
<line x1="{boundary_x:.1f}" y1="{pad_t}" x2="{boundary_x:.1f}" y2="{height-pad_b}" stroke="#334155" stroke-dasharray="6 6"/>
<polyline points="{points(val)}" fill="none" stroke="#0f766e" stroke-width="3"/>
<polyline points="{points(train)}" fill="none" stroke="#7c3aed" stroke-width="2" opacity="0.55"/>
<circle cx="{sx(xs[-1]):.1f}" cy="{sy(val[-1]):.1f}" r="6" fill="#dc2626"/>
<text x="{width-pad_r-10}" y="60" text-anchor="end" font-family="Arial" font-size="16" fill="#0f766e">validation loss</text>
<text x="{width-pad_r-10}" y="82" text-anchor="end" font-family="Arial" font-size="16" fill="#7c3aed">train loss at eval</text>
<text x="{width-pad_r-10}" y="104" text-anchor="end" font-family="Arial" font-size="16" fill="#dc2626">final val {val[-1]:.4f}</text>
<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="16">Total tokens seen (billions)</text>
<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-family="Arial" font-size="16">Loss</text>
</svg>'''
    svg_path.write_text(svg, encoding="utf-8")
    print(f"PLOT_SVG_FALLBACK svg={svg_path} csv={csv_path} points={len(history)} final_val={val[-1]:.4f} reason={exc}")

