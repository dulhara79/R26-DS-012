"""Clinician provisioning is explicit and avoids unsafe password arguments."""

import io

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import seed_clinician
from clinician_api import verify_password
from db_models import Base, Clinician


def test_password_stdin_requires_minimum_length(monkeypatch):
    monkeypatch.setattr(seed_clinician.sys, "stdin", io.StringIO("correct-horse-research-password\n"))
    assert seed_clinician._password_from_operator(True) == "correct-horse-research-password"
    monkeypatch.setattr(seed_clinician.sys, "stdin", io.StringIO("short\n"))
    with pytest.raises(SystemExit, match="12 characters"):
        seed_clinician._password_from_operator(True)


def test_seed_requires_explicit_database_url(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        seed_clinician.main(["DRTEST", "Dr Test"])
    assert exc.value.code == 2
    assert "DATABASE_URL" in capsys.readouterr().err


def test_password_rotation_is_explicit_and_preserves_role(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setattr(seed_clinician, "SessionLocal", Session)
    monkeypatch.setattr(seed_clinician, "init_db", lambda: None)
    monkeypatch.setattr(seed_clinician, "_password_from_operator", lambda use_stdin: "first-long-password-123")

    assert seed_clinician.main(["DRTEST", "Dr Test", "--role", "doctor"]) == 0
    with Session() as db:
        saved = db.get(Clinician, "DRTEST")
        assert saved.active and saved.role == "doctor"
        assert verify_password("first-long-password-123", saved.password_hash)

    with pytest.raises(SystemExit, match="already exists"):
        seed_clinician.main(["DRTEST", "Dr Test Updated"])
    monkeypatch.setattr(seed_clinician, "_password_from_operator", lambda use_stdin: "second-long-password-456")
    assert seed_clinician.main(["DRTEST", "Dr Test", "--rotate-password"]) == 0
    with Session() as db:
        saved = db.get(Clinician, "DRTEST")
        assert saved.role == "doctor"
        assert verify_password("second-long-password-456", saved.password_hash)
        assert not verify_password("first-long-password-123", saved.password_hash)
    engine.dispose()
