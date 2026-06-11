---
name: poly-memoization-design
description: autosi の多項式区間計算メモ化の設計根拠（forward_si の z 反復最適化）
metadata:
  type: project
---

autosi の `inference()` は sicore 経由で `forward_si(a, b, z)` を多数の z で繰り返し呼ぶ。**1つの p-value 計算中は選択直線 a, b が固定で z だけ変化する**ため、各計算地点で生成される多項式係数は呼び出し間で不変、かつ多項式の根（固有値）は z 不変。

これを利用した3層メモ化を [autosi/poly_memo.py](../poly_memo.py) に実装済み:
- レベルA: 根（`np.linalg.eigvals`）を係数ハッシュキーでキャッシュ
- レベルB（地点別）: iso-sign 区間 `[lo,hi]` に次の z が含まれれば計算スキップ。符号別に保持。「z を含む最下位コンポーネントだけ再計算」の効果を地点別に実現
- レベルC（実行全体スキップ）: 前回の最終(最小)累積区間に z が入れば forward_si を丸ごとスキップ。ただし標準 parametric 探索は z を境界外へジャンプするため**実質 0% ヒット**（lasso で実測）。他探索モード用の保険として保持

**Why:** 固有値計算が支配的コスト。係数ハッシュキーなら a/b 変化で自動的に別キーになり誤再利用しない。レベルCの「コンポーネントjから再開」完全形は中間状態replayエンジンが必要で未実装、計算削減はA/Bが地点別に達成済み。
**How to apply:** p-value 毎に `inference()` 開始時 `poly_memo.reset()` 必須（メモリ有界化＋汚染防止）。オン/オフは `inference(memoize=)` か `si.set_memoization()`。lasso で約1.76倍・p-value 完全一致を確認済み。深層学習など領域数が指数的な問題ではキャッシュキー数が `z区間数×地点数` で膨張するため LRU 上限の追加が前提。
