from __future__ import annotations

from sicore import RealSubset


class IntervalTracker:
    """グローバルなintervals管理用のシングルトンクラス"""

    _instance = None
    _intervals = None
    _z = None

    @classmethod
    def get_instance(cls) -> "IntervalTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self, z: float | None = None) -> None:
        """intervalsとzをリセット"""
        self._intervals = RealSubset([[-float("inf"), float("inf")]])
        self._z = z

    def get(self) -> RealSubset:
        """現在のintervalsを取得"""
        if self._intervals is None:
            self.reset()
        return self._intervals

    def get_z(self) -> float | None:
        """現在のzを取得"""
        return self._z

    def set_z(self, z: float) -> None:
        """zを設定"""
        self._z = z

    def update(self, new_intervals: RealSubset) -> RealSubset:
        """intervalsを更新（積集合）"""
        if self._intervals is None:
            self._intervals = new_intervals
        else:
            self._intervals = self._intervals & new_intervals
        return self._intervals

    def get_state(self) -> dict:
        """現在の状態（intervalsとz）を取得"""
        return {"intervals": self._intervals, "z": self._z}
