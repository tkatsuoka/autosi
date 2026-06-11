# autosi 実験

`autosi`（自動 selective inference ライブラリ）を用いて、特徴選択3手法
（marginal screening / lasso / cv_r2_lasso）の **type I error rate (FPR)** と
**検出力 (power)** を、iid・correlated データで評価する実験一式。

PBS (qsub) クラスタでの分散実行を前提とし、各手法・条件の p 値と計算時間を
CSV に保存して折れ線グラフを生成する。

## ファイル構成

```
src/
├── synthetic_fpr.sh     FPR実験ランチャ兼ジョブ本体 (サンプル数 n を振る)
├── synthetic_power.sh   power実験ランチャ兼ジョブ本体 (信号Δ を振る)
├── run_experiment.py    1ジョブ分の計算本体 (p値と計算時間を CSV 出力)
├── environment.py       スレッド数=1 固定・シード設定 (他コアを奪わない)
├── data.py              データ生成 (iid / correlated, null / power)
├── methods.py           3手法の SI アルゴリズム
├── plot_results.py      FPR の折れ線グラフ (横軸=サンプル数)
├── plot_power.py        power の折れ線グラフ (横軸=Δ)
└── merge_results.py     分割CSVの統合・読み込みヘルパ
```

## セットアップ

```bash
uv sync
uv add pandas matplotlib tqdm   # 未追加の場合
```

## 実験の投入 (ログインノードで `bash` 実行 → 内部で qsub)

ランチャは計算せず `qsub` を投げるだけなので、ログインノードで実行してよい。

### FPR (type I error rate)

サンプル数 n ∈ {100, 200, 300, 400} を振り、null データで p 値を計算する。

```bash
# 手法ごとに投入 (-m を変える)
bash src/synthetic_fpr.sh -m lasso              -d iid
bash src/synthetic_fpr.sh -m marginal_screening -d iid
bash src/synthetic_fpr.sh -m cv_r2_lasso        -d iid
```

オプション:

| フラグ | 意味 | 既定 |
|--------|------|------|
| `-m` | 手法 (marginal_screening / lasso / cv_r2_lasso) | lasso |
| `-d` | データ種別 (iid / correlated) | iid |
| `-N` | サンプル数のリスト (横軸) | "100 200 300 400" |
| `-n` | 分割数 (= ジョブ数) | 10 |
| `-p` | 1ジョブが計算する p 値の個数 | 1000 |
| `-w` | 1ジョブ内の並列ワーカー数 | 96 |
| `-q` | キュー名 | cpu-medium |

### power (検出力)

信号 Δ ∈ {0.1, 0.2, 0.3, 0.4} を振り、`y = Xβ + ε`（先頭 k 個の係数を Δ）で
p 値を計算する。

```bash
bash src/synthetic_power.sh -m lasso              -d iid
bash src/synthetic_power.sh -m marginal_screening -d iid
bash src/synthetic_power.sh -m cv_r2_lasso        -d iid
```

オプション (FPR と共通のものに加えて):

| フラグ | 意味 | 既定 |
|--------|------|------|
| `-s` | Δ のリスト (横軸) | "0.1 0.2 0.3 0.4" |
| `-k` | 信号を載せる先頭特徴量の数 | 3 |

## 進捗確認

```bash
qstat -u $USER            # 自分のジョブ状況
```

## 可視化 (全ジョブ完了後)

出力フォルダを渡すだけ（中の `part_*.csv` を再帰的に読み込む。統合は不要）。
(手法 × データ種別) の組ごとに別ファイルで折れ線グラフを出力する。

```bash
# FPR: result/fpr_<method>_<data_type>.pdf
uv run python src/plot_results.py --csv result/fpr

# power: result/power_<method>_<data_type>.pdf
uv run python src/plot_power.py --csv result/power
```

特定条件だけ描くこともできる:

```bash
uv run python src/plot_results.py --csv result/fpr/lasso_iid
```

## 出力CSVの列

| 列 | 内容 |
|----|------|
| method | 手法名 |
| data_type | iid / correlated |
| seed | 乱数シード (= 試行ID) |
| p_value | 選択的 p 値 |
| time | p 値の計算時間 [秒] |
| n_samples | サンプル数 |
| n_features | 特徴量数 |
| signal | 信号の傾き Δ (0 = null) |
| n_signal | 信号を載せた特徴量数 |

## 論文設定との対応

Shiraishi et al. (2023) の SFS (selective z-test) に合わせて既定値を設定済み:

- 特徴量数 p = 10（`--n_features` 既定 10）
- 信号特徴量数 = 5（`--n_signal` 既定 5、β の先頭5要素を Δ）
- FPR: n ∈ {100, 200, 300, 400}（`synthetic_fpr.sh -N`）
- power: Δ ∈ {0.1, 0.2, 0.3, 0.4}, n = 200 固定（`--n_samples` 既定 200）

注: 選択アルゴリズム自体は論文の forward stepwise (K=5) ではなく、本実験の
3手法（各 1 変数を選択）である点が論文と異なる。データ生成設定のみ論文準拠。
