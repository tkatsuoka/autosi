"""合成データ実験の図の共通スタイル定義。

plot_fpr / plot_power で図サイズ・フォント・凡例の見た目を揃えるための
定数と、全図共通の「凡例だけの PDF」を出力するヘルパーを置く。
各図には凡例を描かず、論文では legend.pdf を 1 つだけ並べて使う。
"""

import numpy as np
import matplotlib.pyplot as plt

# 図サイズは黄金比 (幅 4 インチ基準)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
FIGSIZE = (4, 4 / GOLDEN_RATIO)
DPI = 100
FONT_SIZE = 12

# 凡例ラベル。描画順 = 色サイクル順 (C0, C1, ...) で plot_fpr / plot_power と
# 一致させること (naive は power の図には描かれないが凡例には含める)
LABELS = ["proposed", "w/o-pp", "bonferroni", "data splitting", "naive"]


def save_legend(path):
    """全手法を横一列に並べた凡例だけの PDF を出力する。"""
    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=(10, 0.5), dpi=DPI)
    handles = [
        plt.Line2D([], [], color=f"C{i}", marker="x", linestyle="-",
                   label=label)
        for i, label in enumerate(LABELS)
    ]
    fig.legend(handles=handles, ncol=len(LABELS), loc="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {path}")
