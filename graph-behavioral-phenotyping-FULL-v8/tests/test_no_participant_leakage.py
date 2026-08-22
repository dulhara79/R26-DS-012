import numpy as np
from training.cross_validation import participant_grouped_splits


class _Y:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class Dummy:
    def __init__(self, uid, y):
        self.uid = uid
        self.y = _Y(y)


def test_participant_never_crosses_train_test():
    data = []
    for i in range(20):
        for _ in range(3):
            data.append(Dummy(f"u{i:02d}", i % 2))

    groups = np.array([d.uid for d in data])

    for train_idx, test_idx in participant_grouped_splits(
        data,
        n_splits=5,
        random_state=42,
    ):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
