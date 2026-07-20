"""検出力 (power) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: 信号の傾き Δ、縦軸: power (= p 値 < α の割合)。
(手法, データ種別) の組ごとに 2 枚出力する:
  conditional_power_<method>_<data_type>.pdf  条件付き power
                                              (= P(棄却 | 信号特徴量を選択))
  marginal_power_<method>_<data_type>.pdf     周辺 power
                                              (= P(信号特徴量を選択 かつ 棄却))
proposed / w/o-pp / bonferroni / data splitting の 4 曲線を重ねて描く
(色は FPR プロットと対応。naive は FPR を制御しない invalid な手法の
ため power には描かない)。無条件 power は「選択に半分のデータしか
使えない」という DS の弱点も反映した end-to-end の検出力比較になる。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/synthetic/plot_power.py --csv result/power/lasso_iid_d0.1
  python src/synthetic/plot_power.py --csv result/power
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group
from src.common.plot_style import FIGSIZE, DPI, FONT_SIZE, line_style, save_legend


def _power_curve(sub, p_col, flag_col, alpha, n_pvalues, label, method, data_type):
    """1 本分の power 曲線 (Δ -> 棄却率) を集計して返す。

    実際に信号が載った特徴量が選ばれた試行のみで power を測る。
    null 特徴量 (真の係数 0) が選ばれて棄却された試行は Type I error なので、
    power の分母・分子どちらからも除外する (= 条件付き検出力)。
    """
    if flag_col in sub.columns:
        n_total = len(sub)
        sub = sub[sub[flag_col]]
        # 動的モードで目標本数の充足後に SI 推論を省略した行は p 値が
        # 空なので除く (DS 収集専用フェーズの行。除外はシード順の打ち切り
        # と同じで、p 値の値には依存しないためバイアスは生じない)
        sub = sub[sub[p_col].notna()]
        print(f"{method} ({data_type}) [{label}]: 有効 p 値 "
              f"{len(sub)}/{n_total} 件で power を算出")
    else:
        print(f"警告: {method} ({data_type}) に {flag_col} 列が無い (旧形式CSV)。"
              "全試行で power を算出します。run_experiment.py で再計算してください。")

    # 信号選択で絞った後、条件 (Δ) ごとに seed 順で先頭 n_pvalues 件だけ使う
    sub = limit_per_group(sub, ["signal"], n_pvalues)

    # Δ ごとに power = P(p 値 < α | 信号特徴量を選択) を集計
    return (
        sub.assign(reject=sub[p_col] < alpha)
        .groupby("signal")["reject"].mean()
        .sort_index()
    )


def _selection_rate(sub, flag_col):
    """Δ ごとの選択成功率 P(信号特徴量を選択) を全試行から集計して返す。"""
    return sub.groupby("signal")[flag_col].mean().sort_index()


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None,
             unconditional=False):
    """(手法, データ種別) 1 組の power 折れ線グラフ (横軸 Δ) を描く。

    unconditional=True では無条件 power = P(信号特徴量を選択 かつ 棄却)
    = 選択成功率 × 条件付き power を描く。条件付き power は DS の
    「選択に半分のデータしか使えない」弱点を分母から消してしまうため、
    end-to-end の検出力の比較にはこちらが適する。選択成功率は全試行から
    推定する (--n_pvalues は条件付き power 側にのみ適用)。
    """
    # (凡例ラベル, p 値列, 信号選択フラグ列, zorder)。描画順 = 凡例順 =
    # 色サイクル順で plot_fpr と共通 (naive は FPR を制御しない invalid な
    # 手法のため power には描かない)。データ分割法は前半のみで選択する
    # ため、条件付けに使うフラグ列も別 (ds_is_signal)。
    curves = [
        ("proposed", "p_value", "is_signal", 10),
        ("w/o-pp", "oc_p_value", "is_signal", 9),
        ("bonferroni", "bonferroni_p_value", "is_signal", 8),
        ("data splitting", "ds_p_value", "ds_is_signal", 7),
    ]

    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    sigs = sorted(sub["signal"].unique())
    ax.plot(sigs, [alpha] * len(sigs), linestyle="dashed", color="black")
    for label, p_col, flag_col, z in curves:
        # 旧形式 CSV に列が無い場合は飛ばす
        if p_col not in sub.columns:
            continue
        power = _power_curve(sub, p_col, flag_col, alpha, n_pvalues,
                             label, method, data_type)
        if unconditional:
            if flag_col not in sub.columns:
                continue
            # 無条件 power = 選択成功率 × 条件付き power
            power = power * _selection_rate(sub, flag_col)
        ax.plot(power.index, power.values, label=label, zorder=z,
                **line_style(label))

    ax.set_xticks(sigs)
    # 主目盛 0.2 刻み + 補助目盛 (ラベルなし) 0.1 刻み (FPR の図と共通)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1), minor=True)
    ax.set_xlabel("Signal")
    # 縦軸ラベルは conditional / marginal ともに "Power" で統一
    # (どちらの図かはファイル名で区別する)
    ax.set_ylabel("Power")
    # 凡例は各図には描かず、共通の legend.pdf (save_legend) を使う
    prefix = "marginal_power" if unconditional else "conditional_power"
    path = os.path.join(outdir, f"{prefix}_{method}_{data_type}.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/power",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="各条件で使用する p 値の個数 (is_signal で絞った後、"
                             "seed 順の先頭から。既定: 全件)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # (手法, データ種別) の組ごとに条件付き / 無条件の 2 枚ずつ出力
    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues)
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues, unconditional=True)

    # power の全図共通の凡例だけの PDF (naive は power に描かないため除く)
    save_legend(os.path.join(args.outdir, "legend_power.pdf"),
                labels=["proposed", "w/o-pp", "bonferroni", "data splitting"])


if __name__ == "__main__":
    main()
