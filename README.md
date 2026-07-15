# autosi 実験

`autosi`（自動 selective inference ライブラリ）を用いて、特徴選択3手法
（marginal screening / lasso / cv_r2_lasso）を評価する実験一式。

- **合成データ実験** (`src/synthetic/`): type I error rate (FPR) と検出力 (power)
- **実データ実験** (`src/realdata/`): UCI の 8 データセットで棄却率を比較
  (si4pipeline 論文 [Shiraishi et al., 2024] の実データ実験に準拠)

PBS (qsub) クラスタでの分散実行を前提とし、各手法・条件の p 値と計算時間を
CSV に保存して表・グラフを生成する。

## フォルダ構成

```
src/
├── pbs_common.sh            投入スクリプト共通の設定・ヘルパ (ドライバから source)
├── jobs/                    PBS ジョブ本体 (spool 実行のため source しない)
│   ├── synthetic.pbs        run_experiment.py を MYARGS で実行
│   └── realdata.pbs         run_realdata.py を MYARGS で実行
├── common/                  実験共通モジュール
│   ├── environment.py       スレッド数=1 固定・シード設定 (他コアを奪わない)
│   ├── methods.py           3手法の SI アルゴリズム (sklearn 規約の Lasso)
│   └── merge_results.py     分割CSVの統合・読み込みヘルパ
├── synthetic/               合成データ実験
│   ├── data.py              データ生成 (iid / corr, null / power)
│   ├── run_experiment.py    1ジョブ分の計算本体 (p値と計算時間を CSV 出力)
│   ├── fpr.sh               FPR実験ドライバ (サンプル数 n を振る)
│   ├── power.sh             power実験ドライバ (信号Δ を振る)
│   ├── plot_fpr.py          FPR の折れ線グラフ (横軸=サンプル数)
│   └── plot_power.py        power の折れ線グラフ (横軸=Δ)
└── realdata/                実データ実験
    ├── datasets.py          UCI 8データセットの取得・整形 (data/realdata/ にキャッシュ)
    ├── run_realdata.py      1ジョブ分の計算本体 (サブサンプリング -> p値)
    ├── realdata.sh          実データ実験ドライバ
    └── plot_realdata.py     棄却率の表 (論文 Table 1 形式: proposed/oc/ds)
```

旧ファイルからの移動対応: `src/synthetic_fpr.sh` → `src/synthetic/fpr.sh`、
`src/synthetic_power.sh` → `src/synthetic/power.sh`、
`src/plot_results.py` → `src/synthetic/plot_fpr.py`、
`src/{methods,environment,merge_results}.py` → `src/common/`、
`src/data.py` → `src/synthetic/data.py`。

## セットアップ

```bash
uv sync --extra experiments
```

`experiments` エクストラには pandas / matplotlib / tqdm に加えて、
実データ実験の UCI Excel ファイル変換用に openpyxl / xlrd が含まれる。

**注意**: `uv run python ...` を `--extra experiments` なしで実行すると、
uv が extras を外して環境を再同期してしまい pandas 等が消える。
手動実行時は必ず `uv run --extra experiments python ...` とするか、
`.venv/bin/python3` を直接使うこと。PBS の subjob は uv を経由せず
`.venv/bin/python3` を直接使う (計算ノードでの再同期・競合を避けるため)。

## 実験設定 (既定値)

- 特徴量数 p = 20（`--n_features` 既定 20）
- 信号特徴量数 = 5（`--n_signal` 既定 5、β の先頭5要素を Δ）
- FPR: n ∈ {100, 200, 300, 400}（`fpr.sh -N`）
- power: Δ ∈ {0.1, 0.2, 0.3, 0.4}, n = 200 固定（`--n_samples` 既定 200）
- Lasso の正則化は sklearn と同じ規約 (1/(2n))‖y−Xβ‖² + α‖β‖₁（`--alpha` 既定 0.1）
- Bonferroni 補正の比較数は選択パターン総数 2^p
- power は「選ばれた特徴量が真に信号を持つ試行」のみで算出する条件付き検出力
  (`is_signal` 列で判定。null 特徴量の棄却は type I error なので分母からも除外)
- 1 試行が数値誤差などで失敗した場合はスキップして次の試行へ続行し、
  失敗件数をログに出す。既定の `-p` は不足に備えて 1 割多めにしてある
- 比較手法としてデータ分割法 (`ds_p_value`) も毎試行計算する。他手法と
  同じ n 行を前半/後半に2分割し、前半で同じアルゴリズムにより特徴選択、
  後半 (選択と独立) で古典的 z 検定を行う (使用データ量をフェアに揃える)

## PBS への投入方式 (共通)

ドライバ (`fpr.sh` / `power.sh` / `realdata.sh`) は計算せず `qsub` を
投げるだけなので、ログインノードで実行してよい。全作業（条件 × 分割）を
**1 ジョブずつ** `qsub -N <名前> -o eo -e eo -v MYARGS="..." src/jobs/*.pbs`
の形で投入する。キュー・ワーカー数・eo/ の場所などの共通設定は
`src/pbs_common.sh` に集約してある。ジョブ本体 (`src/jobs/*.pbs`) は
spool 実行されるため、`PBS_O_WORKDIR` 起点で `.venv/bin/python3` を
直接呼び、`MYARGS` を Python スクリプトへそのまま渡す。

## 合成データ実験

### FPR (type I error rate)

サンプル数 n を振り、null データ (signal=0) で p 値を計算する。

```bash
# 手法ごとに投入 (-m を変える)
bash src/synthetic/fpr.sh -m lasso              -d iid
bash src/synthetic/fpr.sh -m marginal_screening -d iid
bash src/synthetic/fpr.sh -m cv_r2_lasso        -d iid
```

| フラグ | 意味 | 既定 |
|--------|------|------|
| `-m` | 手法 (marginal_screening / lasso / cv_r2_lasso) | lasso |
| `-d` | データ種別 (iid / corr) | iid |
| `-N` | サンプル数のリスト (横軸) | "100 200 300 400" |
| `-n` | 分割数 (サンプル数ごとのジョブ数) | 10 |
| `-p` | 1ジョブが計算する p 値の個数 (エラー時の不足に備え 1 割の余裕) | 1100 |
| `-w` | 1ジョブ内の並列ワーカー数 | 96 |
| `-q` | キュー名 | cpu-large |

### power (検出力)

信号 Δ を振り、`y = Xβ + ε`（先頭 `-k` 個の係数を Δ）で p 値を計算する。
n は `run_experiment.py` の既定 (200) に固定される。

```bash
bash src/synthetic/power.sh -m lasso              -d iid
bash src/synthetic/power.sh -m marginal_screening -d iid
bash src/synthetic/power.sh -m cv_r2_lasso        -d iid
```

オプション (FPR と共通の `-m -d -n -p -w -q -j` に加えて):

| フラグ | 意味 | 既定 |
|--------|------|------|
| `-s` | Δ のリスト (横軸) | "0.1 0.2 0.3 0.4" |
| `-k` | 信号を載せる先頭特徴量の数 | 5 |

### ローカルで直接実行する場合

```bash
uv run --extra experiments python src/synthetic/run_experiment.py --iter 10 --methods lasso \
    --data_types iid --signal 0.2 --out result/local_test.csv
```

主なオプション: `--iter` 反復回数 / `--workers` 並列数 /
`--methods` `--data_types` カンマ区切りの対象 /
`--n_samples` (既定 200) `--n_features` (既定 20) `--signal` (0=null)
`--n_signal` (既定 5) / `--seed_start` `--seed_end` `--seed_offset`
ジョブ分割用 seed 範囲 / `--alpha` `--max_iter` `--tol` Lasso ハイパラ /
`--K` `--n_lambda` `--lam_lo` `--lam_hi` CV ハイパラ / `--out` 出力先。

### 可視化 (全ジョブ完了後)

出力フォルダを渡すだけ（中の `part_*.csv` を再帰的に読み込む。統合は不要）。
(手法 × データ種別) の組ごとに別ファイルで折れ線グラフを出力する。

```bash
# FPR: result/fpr_<method>_<data_type>.pdf
#      (proposed/oc/naive/bonferroni/data splitting の5本)
uv run --extra experiments python src/synthetic/plot_fpr.py --csv result/fpr

# power: result/power_<method>_<data_type>.pdf
#        (proposed と data splitting の2本。各自の信号選択試行のみで算出)
uv run --extra experiments python src/synthetic/plot_power.py --csv result/power

# 使用する p 値の個数を条件ごとに 1000 件に揃える場合
uv run --extra experiments python src/synthetic/plot_fpr.py --csv result/fpr --n_pvalues 1000
```

共通オプション: `--alpha` 有意水準 (既定 0.05) / `--n_pvalues` 各条件で
使用する p 値の個数 (seed 順の先頭から。既定は全件)。`-p` の 1 割の余裕分を
切り捨てて条件間で件数を揃えるのに使う。power では is_signal で絞った後に
適用され、不足する条件は警告を出してある分だけで集計する。

## 実データ実験

si4pipeline 論文の実データ実験に準拠。UCI の 8 データセット
(CC BY 4.0) から n 行をサブサンプリングして p 値を計算し、
**棄却率 P(p 値 < α)** を (データセット × n) ごとに比較する
(実データは真の信号が未知のため、棄却率を power の指標とする)。

| 名前 | データセット (論文の対応) | 行数 | p |
|------|--------------------------|------|---|
| airfoil | Airfoil Self-Noise (Data1) | 1503 | 5 |
| concrete | Concrete Compressive Strength (Data2) | 1030 | 8 |
| energy_heating | Energy Efficiency, heating load (Data3) | 768 | 8 |
| energy_cooling | Energy Efficiency, cooling load (Data4) | 768 | 8 |
| gas_turbine | Gas Turbine CO and NOx, 目的変数 NOX (Data5) | 36733 | 9 |
| real_estate | Real Estate Valuation (Data6) | 414 | 6 |
| wine_red | Wine Quality, red (Data7) | 1599 | 11 |
| wine_white | Wine Quality, white (Data8) | 4898 | 11 |

各試行の処理: サブサンプリング → X 列標準化・y 標準化 →
σ² を OLS 残差から推定 (自由度 n−p−1) → y の中心化を含むアルゴリズムで
特徴選択 → 選択的 p 値を計算。切片は学習しない代わりに y の中心化を
「アルゴリズム内で」行う (無罰則切片と等価で、autosi が追跡するため valid)。

### 手順

```bash
# 1. データセットの取得 (初回のみ・要ネットワーク。data/realdata/ にキャッシュ)
uv run --extra experiments python src/realdata/datasets.py

# 2. 実験の投入 (ドライバが手順1も自動実行する)
bash src/realdata/realdata.sh -m lasso
bash src/realdata/realdata.sh -m marginal_screening
bash src/realdata/realdata.sh -m cv_r2_lasso

# 3. 集計 (全ジョブ完了後)
uv run --extra experiments python src/realdata/plot_realdata.py --csv result/realdata
```

`realdata.sh` のオプション:

| フラグ | 意味 | 既定 |
|--------|------|------|
| `-m` | 手法 (marginal_screening / lasso / cv_r2_lasso) | lasso |
| `-D` | データセットのリスト | 全8種 |
| `-N` | サブサンプルのサイズ n のリスト | "100 150 200" |
| `-n` | 分割数 ((データセット, n) ごとのジョブ数) | 10 |
| `-p` | 1ジョブが計算する p 値の個数 (10 分割 × 110 = 1100 試行) | 110 |
| `-w` | 1ジョブ内の並列ワーカー数 | 96 |
| `-q` | キュー名 | cpu-large |

ローカルで直接実行する場合 (オプションは `run_experiment.py` とほぼ共通、
`--data_types` の代わりに `--datasets`、`--signal` 系はなし):

```bash
uv run --extra experiments python src/realdata/run_realdata.py --iter 10 --methods lasso \
    --datasets wine_red --n_samples 100 --out result/local_realdata.csv
```

`plot_realdata.py` は α=0.05 (変更は `--alpha`) の棄却率を
`realdata_summary.csv` (縦持ち) と、手法ごとの
`realdata_table_<method>.csv` (行 = n, 列 = データセット,
セル = proposed/oc/ds の論文 Table 1 形式) に出力する。
`--n_pvalues` で各条件 (手法, n, データセット) が使用する p 値の個数を
seed 順の先頭から揃えられる (既定は全件)。

## 出力CSVの列

### 合成データ (`run_experiment.py`)

| 列 | 内容 |
|----|------|
| method | 手法名 |
| data_type | iid / corr |
| seed | 乱数シード (= 試行ID) |
| p_value | 選択的 p 値 |
| oc_p_value | over-conditioning の p 値 (比較用) |
| naive_p_value | 選択を無視した naive p 値 (比較用) |
| bonferroni_p_value | Bonferroni 補正 p 値 (比較数 = 2^p) |
| time | p 値の計算時間 [秒] |
| n_samples | サンプル数 |
| n_features | 特徴量数 |
| signal | 信号の傾き Δ (0 = null) |
| n_signal | 信号を載せた特徴量数 |
| selected_index | 観測データで選ばれた特徴量の番号 |
| is_signal | 選ばれた特徴量が真に信号を持つか (power の条件付けに使用) |
| ds_p_value | データ分割法の p 値 (前半で選択、後半で z 検定) |
| ds_selected_index | データ分割法 (前半のみ) で選ばれた特徴量の番号 |
| ds_is_signal | データ分割法の選択が真に信号を持つか (power の条件付けに使用) |

### 実データ (`run_realdata.py`)

合成データとの差分: `data_type / signal / n_signal / is_signal / ds_is_signal`
の代わりに (真の信号が未知のため is_signal 系はなし)

| 列 | 内容 |
|----|------|
| dataset | データセット名 |
| n_features | 標準化後の特徴量数 (定数列は除外されるため変動し得る) |
| sigma2 | OLS 残差から推定した分散 |

## 進捗確認・トラブルシュート

```bash
qstat -u $USER            # 自分のジョブ状況 (ジョブ名で条件が分かる)
ls eo/                    # PBS の標準出力/エラーログ (ジョブごとに 1 ファイル)
```

結果 CSV は各ジョブが全試行を計算し終えた時点で書かれる (lasso で数十分〜)。
ログ (eo/) も多くの PBS 設定ではジョブ終了時にコピーされるため、
走行中の生死確認は `qstat` を使う。ジョブがすぐ終わる・結果 CSV が
できない場合は `eo/<ジョブ名>.o<ID>` を確認する。Python の import エラーが
出ている場合は `uv sync --extra experiments` を実行し直してから再投入する。
