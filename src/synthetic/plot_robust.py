"""ロバスト性実験 (robust.sh) の結果を手法ごとに折れ線グラフ化する。

proposed の type I error rate (= p 値 < α の割合) を描く。
- var  : 分散を同じデータから推定した実験。横軸 n、線は有意水準 α ごと
         -> result/robust_var_<method>.pdf
- noise: 非ガウスノイズの実験。横軸 N(0,1) からの 1-Wasserstein 距離、
         線はノイズ分布族ごと。α ごとに 1 枚 (α=0.05 なら a05)
         -> result/robust_noise_<method>_a05.pdf
凡例は各図には描かず、legend_robust_var.pdf / legend_robust_noise.pdf を共通で使う。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/synthetic/plot_robust.py --csv result/robust --n_pvalues 1000
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group
from src.common.plot_style import FIGSIZE, DPI, FONT_SIZE, MARKERS, save_legend
from src.synthetic.data import NOISES

# 描画する有意水準 (同じ p 値から α ごとの棄却率を計算する)
ALPHAS = [0.01, 0.05, 0.10]
# 非ガウスノイズの分布族 (描画順 = 凡例順 = 色の順)
FAMILIES = [nz for nz in NOISES if nz != "gaussian"]
# 縦軸の上限。var は全 α を 1 枚に描くので固定値、noise は α の倍数
VAR_YMAX = 0.2
NOISE_YMAX_RATIO = 4.0


def _alpha_label(alpha):
    return rf"$\alpha={alpha:.2f}$"


def _indexed_style(i):
    """i 番目の系列の書式。plot_style と同じく色とマーカーで二重に区別する。"""
    return {"marker": MARKERS[i], "color": f"C{i}", "linestyle": "-"}


def alpha_style(label):
    """有意水準の凡例ラベルから線の書式を引く (図と凡例で共通)。"""
    return _indexed_style([_alpha_label(a) for a in ALPHAS].index(label))


def family_style(label):
    """ノイズ分布族の名前から線の書式を引く (図と凡例で共通)。"""
    return _indexed_style(FAMILIES.index(label))


def _rejection_rate(sub, group_col, alpha):
    """group_col の値ごとの proposed の棄却率 P(p 値 < α)。"""
    return (
        sub.assign(reject=sub["p_value"] < alpha)
        .groupby(group_col)["reject"].mean()
        .sort_index()
    )


def _new_axes():
    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    return fig, fig.add_subplot(111)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved {path}")


def plot_var(sub, method, outdir, n_pvalues=None):
    """分散推定の実験: 横軸 n、α ごとの type I error rate を 1 枚に描く。"""
    sub = limit_per_group(sub, ["n_samples"], n_pvalues)
    fig, ax = _new_axes()
    for i, alpha in enumerate(ALPHAS):
        style = _indexed_style(i)
        # 各 α の基準線は同じ色の破線
        ax.axhline(alpha, linestyle="dashed", color=style["color"], linewidth=1.0)
        rate = _rejection_rate(sub, "n_samples", alpha)
        ax.plot(rate.index, rate.values, **style)

    ax.set_xticks(sorted(sub["n_samples"].unique()))
    ax.set_ylim(0.0, VAR_YMAX)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Type I Error Rate")
    _save(fig, os.path.join(outdir, f"robust_var_{method}.pdf"))


def plot_noise(sub, method, alpha, outdir, n_pvalues=None):
    """非ガウスノイズの実験: 横軸 Wasserstein 距離、分布族ごとの線を描く。"""
    sub = limit_per_group(sub, ["noise", "distance"], n_pvalues)
    fig, ax = _new_axes()
    ax.axhline(alpha, linestyle="dashed", color="black")
    for fam in FAMILIES:
        fam_sub = sub[sub["noise"] == fam]
        if fam_sub.empty:
            continue
        rate = _rejection_rate(fam_sub, "distance", alpha)
        ax.plot(rate.index, rate.values, **family_style(fam))

    ax.set_xticks(sorted(sub["distance"].unique()))
    ax.set_ylim(0.0, NOISE_YMAX_RATIO * alpha)
    ax.set_xlabel("Wasserstein Distance")
    ax.set_ylabel("Type I Error Rate")
    # ファイル名に '.' を含めない (LaTeX の拡張子判定と衝突するため)
    tag = f"a{round(alpha * 100):02d}"
    _save(fig, os.path.join(outdir, f"robust_noise_{method}_{tag}.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/robust",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="各条件で使用する p 値の個数 (seed 順の先頭から。"
                             "既定: 全件)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # var / noise の判別は CSV に記録した設定列で行う
    var_df = df[df["estimate_var"].fillna(False).astype(bool)]
    noise_df = df[df["noise"] != "gaussian"]

    for method, sub in var_df.groupby("method"):
        plot_var(sub, method, args.outdir, n_pvalues=args.n_pvalues)
    for method, sub in noise_df.groupby("method"):
        for alpha in ALPHAS:
            plot_noise(sub, method, alpha, args.outdir, n_pvalues=args.n_pvalues)

    # 全図共通の凡例だけの PDF
    if not var_df.empty:
        save_legend(os.path.join(args.outdir, "legend_robust_var.pdf"),
                    [_alpha_label(a) for a in ALPHAS], style_fn=alpha_style)
    if not noise_df.empty:
        save_legend(os.path.join(args.outdir, "legend_robust_noise.pdf"),
                    FAMILIES, style_fn=family_style)


if __name__ == "__main__":
    main()
