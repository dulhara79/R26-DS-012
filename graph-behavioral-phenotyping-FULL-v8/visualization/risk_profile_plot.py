"""Legacy visualization guard."""


def plot_risk_profile(*args, **kwargs):
    raise RuntimeError(
        "Hourly risk-profile plots are not part of final Component 2 v8."
    )
