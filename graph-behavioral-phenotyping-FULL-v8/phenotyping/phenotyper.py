"""Legacy phenotype module.

K-Means graph-embedding phenotypes were exploratory and are not presented as a
validated final-v8 clinical output.
"""

def cluster_phenotypes(*args, **kwargs):
    raise RuntimeError(
        "Phenotype clustering is retained as historical work only and is not "
        "part of the final validated v8 output."
    )
