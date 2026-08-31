from anxiety_risk.model import CATEGORY_LEVELS


def test_reference_categories_are_human_readable_baselines():
    assert CATEGORY_LEVELS["gender"][0] == "female"
    assert CATEGORY_LEVELS["edu"][0] == "bachelor's degree"
    assert CATEGORY_LEVELS["smoke"][0] == "never smokes"
    assert CATEGORY_LEVELS["drink"][0] == "never drinks"
