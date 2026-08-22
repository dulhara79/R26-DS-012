from evaluation.permutation_test import participant_label_permutation


class _Y:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class Dummy:
    def __init__(self, uid, y):
        self.uid = uid
        self.y = _Y(y)


def test_permutation_preserves_participant_label_counts():
    data = []
    for i in range(10):
        label = 1 if i < 3 else 0
        for _ in range(2):
            data.append(Dummy(f"u{i}", label))

    mapping = participant_label_permutation(data, seed=7)
    assert sum(mapping.values()) == 3
    assert len(mapping) == 10
