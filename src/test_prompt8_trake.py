import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from src.tasks.task3_trake import align_events_time_aware


def test_respects_time_gap_not_row_count():
    """
    Hai video co cung so keyframe nhung mat do thoi gian khac nhau: video
    'day dac' (0.1s/keyframe) va video 'thua' (2.0s/keyframe). Voi cung
    min_gap_seconds=1.0, ket qua phai TON TRONG khoang cach GIAY THUC,
    khong phai so dong ma tran.
    """
    T, N = 20, 2
    scores = np.zeros((T, N), dtype=np.float64)
    # Event 1 co peak that su o keyframe 2, event 2 co peak "gia" rat gan
    # (keyframe 3) de kiem tra DP co bi ep chon no hay khong khi vi pham gap.
    scores[2, 0] = 5.0
    scores[3, 1] = 4.9   # qua gan event 1 neu video day dac
    scores[15, 1] = 4.8  # ung vien hop le, cach xa hon

    dense_pts = np.arange(T) * 0.1          # video day dac: 0.1s/keyframe
    sparse_pts = np.arange(T) * 2.0         # video thua: 2.0s/keyframe

    aligned_dense, feasible_dense, _, _, _ = align_events_time_aware(
        scores, dense_pts, min_gap_seconds=1.0
    )
    aligned_sparse, feasible_sparse, _, _, _ = align_events_time_aware(
        scores, sparse_pts, min_gap_seconds=1.0
    )

    # Video day dac: keyframe 3 chi cach keyframe 2 la 0.1s < 1.0s -> BAT
    # BUOC phai chon keyframe 15 (~1.3s sau) cho event 2, du diem thap hon.
    assert aligned_dense[0] == 2
    assert aligned_dense[1] == 15
    assert feasible_dense is True

    # Video thua: keyframe 3 da cach keyframe 2 toi 2.0s >= 1.0s -> duoc
    # phep chon ngay keyframe 3 (diem cao hon, gan nghia hon).
    assert aligned_sparse[0] == 2
    assert aligned_sparse[1] == 3
    assert feasible_sparse is True

    print("[PASS] time-aware gap (khong phai row-count gap)")


def test_monotonic_order_enforced():
    T, N = 10, 3
    scores = np.random.RandomState(0).rand(T, N)
    # Ep peak "tu nhien" cua ca 3 event nam gan nhau/dao thu tu de kiem tra
    # DP van tra ve chuoi tang dan.
    scores[:, 0] = 0.0
    scores[:, 1] = 0.0
    scores[:, 2] = 0.0
    scores[1, 0] = 1.0
    scores[1, 1] = 1.0  # trung vi tri voi event 1 - phai bi day ra xa hon
    scores[1, 2] = 1.0

    pts = np.arange(T) * 1.0
    aligned, feasible, _, _, _ = align_events_time_aware(scores, pts, min_gap_seconds=1.0)

    assert aligned == sorted(aligned)
    assert len(set(aligned)) == N  # khong trung keyframe giua cac event
    print("[PASS] monotonic order voi peak trung nhau")


def test_infeasible_falls_back_without_crash():
    # T < N -> khong the align, phai fallback linspace, khong raise loi.
    T, N = 2, 5
    scores = np.random.RandomState(1).rand(T, N)
    pts = np.arange(T) * 1.0
    aligned, feasible, dp_score, _, _ = align_events_time_aware(
        scores, pts, min_gap_seconds=1.0
    )
    assert len(aligned) == N
    assert feasible is False
    assert dp_score == 0.0
    print("[PASS] fallback khi T < N khong crash")


def main():
    test_respects_time_gap_not_row_count()
    test_monotonic_order_enforced()
    test_infeasible_falls_back_without_crash()
    print("=" * 60)
    print("PROMPT 8 TRAKE ALIGNMENT TESTS: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()