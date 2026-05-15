import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, 'r') as f:
        rows = json.load(f)
    return sorted(rows, key=lambda row: (row['date'], row['mu_comm'], row['bar_ls_paper'], row['comm_snr_db']))


def write_csv(rows, path):
    fieldnames = [
        'date',
        'name',
        'comm_snr_db',
        'mu_comm',
        'psnr',
        'bar_ls_paper',
        'avg_rate_level',
        'avg_kept_real_symbols',
        'checkpoint',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_series(rows):
    series = {}
    for row in rows:
        label = f"{row['date']}  μ={row['mu_comm']}"
        series.setdefault(label, []).append(row)
    for label in series:
        series[label] = sorted(series[label], key=lambda row: row['bar_ls_paper'])
    return series


def plot(rows, output_path, title):
    series = build_series(rows)
    colors = {
        '26-04-19  μ=0.0001': '#0b6e4f',
        '26-04-20  μ=0.0005': '#c44536',
        '26-04-21  μ=0.001': '#1f5aa6',
    }

    fig = plt.figure(figsize=(18, 10), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    for label, points in series.items():
        xs = [row['bar_ls_paper'] for row in points]
        ys = [row['psnr'] for row in points]
        color = colors.get(label, None)
        ax.plot(xs, ys, marker='o', linewidth=2.2, markersize=7, label=label, color=color)
        for row in points:
            ax.annotate(
                f"SNR {int(row['comm_snr_db'])}",
                (row['bar_ls_paper'], row['psnr']),
                textcoords='offset points',
                xytext=(6, 6),
                fontsize=8,
                color=color,
            )

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(r'Paper Communication Cost $\bar{l}_s$', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=10, frameon=True)

    ax_table.axis('off')
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row['date'],
                f"{row['mu_comm']:.0e}",
                int(row['comm_snr_db']),
                f"{row['psnr']:.2f}",
                f"{row['bar_ls_paper']:.1f}",
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=['Date', 'μ', 'SNR', 'PSNR', r'$\bar{l}_s$'],
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.38)
    ax_table.set_title('Rate-Quality Summary', fontsize=14, pad=14)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_png', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--title', default='SemCom Rate-Quality Summary (2026-04-19 / 20 / 21)')
    args = parser.parse_args()

    rows = load_rows(args.input_json)
    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_csv)
    plot(rows, output_png, args.title)


if __name__ == '__main__':
    main()
