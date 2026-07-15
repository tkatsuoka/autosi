#!/bin/bash
# PBS 共通ヘルパ。ランチャ (synthetic/*.sh, realdata/*.sh) から source して使う。
#   - detect_queue_run_limit : キューの「ユーザーあたり同時実行上限」を qstat から取得
#   - resolve_throttle       : -j 明示指定 > 自動検出 > 空(=キュー設定任せ) を解決
#   - encode_list / decode_list : 空白区切りリストを qsub -v で安全に渡すための符号化
#
# 計算ノード (subjob) 側では source 不要 (このファイルに依存しない)。

# キューに設定されたユーザーあたり同時実行ジョブ数の上限を標準出力に返す。
# 取得・解析できなければ何も出力しない (呼び出し側で空判定)。
detect_queue_run_limit() {
    local queue=$1 out n pat
    out=$(qstat -Qf "$queue" 2>/dev/null) || return 0
    # 空白・タブを除去して属性を拾いやすくする (継続行の折り返し対策も兼ねる)。
    out=$(printf '%s' "$out" | tr -d ' \t')
    # 代表的な属性名を優先順に探す。
    #   max_run = [u:PBS_GENERIC=20]  /  max_user_run = 20  など
    for pat in 'max_run=\[u:PBS_GENERIC=' 'max_user_run=' 'max_run_soft=\[u:PBS_GENERIC='; do
        n=$(printf '%s\n' "$out" | grep -oE "${pat}[0-9]+" | grep -oE '[0-9]+$' | head -1) || n=""
        if [ -n "$n" ]; then printf '%s' "$n"; return 0; fi
    done
    return 0
}

# 実際に使う同時実行スロット数を決定して標準出力に返す。
#   $1: -j で明示指定された値 (空可)   $2: キュー名
# 明示指定があればそれを、無ければ自動検出値を、いずれも無ければ空文字を返す。
# 空文字の場合は %throttle を付けず、キュー設定の同時実行上限にそのまま従う。
resolve_throttle() {
    local explicit=$1 queue=$2
    if [ -n "$explicit" ]; then printf '%s' "$explicit"; return 0; fi
    detect_queue_run_limit "$queue"
}

# 空白区切りリストを qsub -v で安全に渡すため ':' 区切りへ符号化 / 復号する。
# (-v は値中の空白を扱いにくいため)
encode_list() { printf '%s' "${1// /:}"; }
decode_list() { printf '%s' "${1//:/ }"; }
