from __future__ import annotations

from pathlib import Path

from ..models import SourceConfig
from .base import SourceConnector
from .http_page import HttpPageConnector
from .local_files import LocalFilesConnector
from .ncbi import PmcConnector, PubMedConnector
from .nice import NiceSyndicationConnector


def build_connector(
    source: SourceConfig,
    project_root: Path,
    user_agent: str,
    timeout_seconds: float,
) -> SourceConnector:
    if source.connector == "http_page":
        return HttpPageConnector(user_agent, timeout_seconds)
    if source.connector == "pubmed":
        return PubMedConnector(user_agent, timeout_seconds)
    if source.connector == "pmc":
        return PmcConnector(user_agent, timeout_seconds)
    if source.connector == "nice_syndication":
        return NiceSyndicationConnector(user_agent, timeout_seconds)
    if source.connector == "local_files":
        return LocalFilesConnector(project_root)
    raise ValueError(f"Unsupported connector: {source.connector}")
