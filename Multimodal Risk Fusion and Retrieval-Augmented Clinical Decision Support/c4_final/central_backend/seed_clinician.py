"""Provision a clinician on the intended Central Backend database.

Never accept the password as a positional CLI argument: it could leak through
process lists and shell history. Use interactive input or a protected stdin.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from clinician_api import hash_password
from db_models import Clinician, SessionLocal, init_db


def _password_from_operator(use_stdin: bool) -> str:
    if use_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Clinician password: ")
        confirm = getpass.getpass("Repeat password: ")
        if password != confirm:
            raise SystemExit("Passwords did not match; account was not changed.")
    if len(password) < 12:
        raise SystemExit("Password must contain at least 12 characters; account was not changed.")
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clinician_id", help="Exact case-sensitive clinician ID")
    parser.add_argument("display_name")
    parser.add_argument("--role", choices=("clinician", "doctor"), default=None)
    parser.add_argument("--password-stdin", action="store_true", help="Read password from protected stdin")
    parser.add_argument("--rotate-password", action="store_true", help="Explicitly rotate an existing account password")
    args = parser.parse_args(argv)

    if not args.clinician_id.strip() or args.clinician_id != args.clinician_id.strip():
        parser.error("clinician_id must be nonempty and have no surrounding whitespace")
    if not args.display_name.strip():
        parser.error("display_name must be nonempty")
    if not os.getenv("DATABASE_URL", "").strip():
        parser.error("DATABASE_URL must identify the intended database; refusing default SQLite")

    init_db()
    with SessionLocal() as db:
        existing = db.get(Clinician, args.clinician_id)
        if existing is not None and not args.rotate_password:
            raise SystemExit("Clinician account already exists; use --rotate-password to change it.")
        password = _password_from_operator(args.password_stdin)
        if existing is None:
            db.add(Clinician(
                clinician_id=args.clinician_id,
                display_name=args.display_name.strip(),
                role=args.role or "clinician",
                password_hash=hash_password(password),
                active=True,
            ))
        else:
            existing.display_name = args.display_name.strip()
            if args.role is not None:
                existing.role = args.role
            existing.password_hash = hash_password(password)
            existing.active = True
        db.commit()

    print(f"{'rotated' if existing is not None else 'created'} clinician {args.clinician_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
