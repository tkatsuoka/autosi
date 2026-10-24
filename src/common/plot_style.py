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
# 一致させること。naive は末尾 (C4) なので、naive を含まない power 用の
# 凡例でも先頭 4 手法の色は FPR 側と同一になる
LABELS = ["proposed", "w/o-pp", "bonferroni", "ds", "naive"]
# 手法ごとのマーカー (LABELS と同順)。曲線が重なっても判別でき、白黒印刷や
# 色覚多様性にも頑健なように色とマーカーで二重に区別する
MARKERS = ["o", "s", "^", "D", "x"]


def line_style(label):
    """手法名からマーカーと色を引く (図と凡例で共通に使う)。"""
    i = LABELS.index(label)
    return {"marker": MARKERS[i], "color": f"C{i}", "linestyle": "-"}


def save_legend(path, labels=None, style_fn=line_style):
    """手法を横一列に並べた凡例だけの PDF を出力する。

    labels を省略すると全手法 (LABELS)。power 用など一部の手法だけの
    凡例を出す場合は LABELS の先頭からの部分列を渡す (色対応を保つため)。
    手法以外 (有意水準・ノイズ分布など) の凡例は、ラベルから線の書式を
    返す style_fn を渡す。
    """
    labels = LABELS if labels is None else labels
    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=(10, 0.5), dpi=DPI)
    handles = [
        plt.Line2D([], [], label=label, **style_fn(label))
        for label in labels
    ]
    fig.legend(handles=handles, ncol=len(labels), loc="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {path}")
