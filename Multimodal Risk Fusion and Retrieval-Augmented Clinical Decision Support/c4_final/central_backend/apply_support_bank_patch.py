"""One-shot patcher for the TC-WPN support bank integration. Safe: writes
only where confident; prints WARNING and skips otherwise."""
import re

def patch_db_models():
    path = "db_models.py"
    content = open(path, encoding="utf-8").read()
    if "class SupportBankNote" in content:
        print("[db_models.py] SupportBankNote already present — skipping")
        return
    changed = False
    m = re.search(r"from sqlalchemy import\s*(\(.*?\)|[^\n]+)", content, re.S)
    if m:
        block = m.group(0)
        if re.search(r"\bText\b", block):
            print("[db_models.py] Text already imported")
        else:
            if block.strip().endswith(")"):
                new_block = block[:block.rfind(")")].rstrip()
                if new_block.endswith(","):
                    new_block = new_block[:-1]
                new_block += ", Text)"
            else:
                new_block = block.rstrip() + ", Text"
            content = content.replace(block, new_block, 1)
            changed = True
            print("[db_models.py] Added Text import")
    else:
        print("[db_models.py] WARNING: sqlalchemy import not found — add `Text` manually")

    class_code = '''

# ── TC-WPN support bank ──────────────────────────────────────────────────────
class SupportBankNote(Base):
    """One labelled reference note used to build TC-WPN's class prototypes."""
    __tablename__ = "support_bank_notes"
    __table_args__ = (
        UniqueConstraint("bank_version", "note_id", name="uq_support_note"),
        Index("ix_support_bank_lookup", "bank_version", "label", "active"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bank_version: Mapped[str] = mapped_column(String(32), index=True)
    note_id: Mapped[str] = mapped_column(String(48))
    label: Mapped[str] = mapped_column(String(16))
    note_text: Mapped[str] = mapped_column(Text)
    days_before_index: Mapped[float] = mapped_column(Float, default=0.0)
    source_subject_id: Mapped[Optional[str]] = mapped_column(String(64))
    provenance: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
'''
    inserted = False
    for anchor in [r"^class Subject\(Base\):", r"^def init_db\("]:
        am = re.search(anchor, content, re.M)
        if am:
            content = content[:am.start()] + class_code.strip("\n") + "\n\n\n" + content[am.start():]
            inserted = True
            changed = True
            print(f"[db_models.py] Inserted SupportBankNote before anchor: {anchor}")
            break
    if not inserted:
        content = content.rstrip() + "\n" + class_code
        changed = True
        print("[db_models.py] No anchor found — appended SupportBankNote to end of file (safe)")

    if changed:
        open(path, "w", encoding="utf-8").write(content)
        print("[db_models.py] written")

def patch_main():
    path = "main.py"
    content = open(path, encoding="utf-8").read()
    changed = False

    m = re.search(r"from db_models import\s*(\(.*?\)|[^\n]+)", content, re.S)
    if m:
        block = m.group(0)
        need = [n for n in ("SupportBankNote", "SessionLocal") if not re.search(rf"\b{n}\b", block)]
        if need:
            if block.strip().endswith(")"):
                idx = block.rfind(")")
                new_block = block[:idx].rstrip()
                if new_block.endswith(","):
                    new_block = new_block[:-1]
                new_block += ", " + ", ".join(need) + ")"
            else:
                new_block = block.rstrip() + ", " + ", ".join(need)
            content = content.replace(block, new_block, 1)
            changed = True
            print(f"[main.py] Added {need} to db_models import")
        else:
            print("[main.py] db_models import already complete")
    else:
        print("[main.py] WARNING: `from db_models import ...` not found — add SupportBankNote, SessionLocal manually")

    if "from support_bank import" not in content:
        insert_text = ('\nfrom support_bank import (SupportBankUnavailable, describe_bank,\n'
                       '                          seed_support_bank, select_support_set)\n'
                       'SUPPORT_BANK_VERSION = __import__("os").getenv("SUPPORT_BANK_VERSION", "synthetic-v1")\n')
        anchor = m.group(0) if m else None
        if anchor and anchor in content:
            content = content.replace(anchor, anchor + insert_text, 1)
        else:
            idx = content.find("app = FastAPI(")
            content = (content[:idx] + insert_text + "\n\n" + content[idx:]) if idx != -1 else (insert_text + content)
        changed = True
        print("[main.py] Added support_bank import + SUPPORT_BANK_VERSION")
    else:
        print("[main.py] support_bank import already present")

    if "seed_support_bank(" not in content or "_sb_n = seed_support_bank" not in content:
        def _seed_after(mm):
            ind = mm.group(1)
            return (mm.group(0) + "\n" +
                    f"{ind}try:\n{ind}    _sb_db = SessionLocal()\n"
                    f"{ind}    _sb_n = seed_support_bank(_sb_db)\n"
                    f"{ind}    if _sb_n: print(f\"[startup] seeded support bank ({{_sb_n}} notes)\")\n"
                    f"{ind}except Exception as _sb_exc:\n"
                    f"{ind}    print(f\"[startup] support bank seed skipped: {{_sb_exc}}\")\n"
                    f"{ind}finally:\n"
                    f"{ind}    try: _sb_db.close()\n{ind}    except Exception: pass")
        new_content, n = re.subn(r"^([ \t]*)init_db\(\)[ \t]*$", _seed_after, content, flags=re.M)
        if n:
            content = new_content
            changed = True
            print(f"[main.py] Wired support-bank seeding after {n} init_db() call(s)")
        else:
            print("[main.py] WARNING: no `init_db()` call found — seed manually")
    else:
        print("[main.py] support bank seeding already wired in")

    old_call = '''    result = mc.call_c3(req.note_text, req.note_type,
                        req.anxiety_support, req.control_support,
                        support_set=req.support_set,
                        note_date=req.note_date,
                        visit_count=req.visit_count,
                        subject_external_id=_external_id(db, subject_id, "c3_clinical_nlp"))'''
    new_call = '''    support_set = req.support_set or None
    support_set_version = None
    if not support_set:
        try:
            support_set = select_support_set(
                db, subject_id=subject_id, bank_version=SUPPORT_BANK_VERSION)
            support_set_version = SUPPORT_BANK_VERSION
        except SupportBankUnavailable as exc:
            result = mc.ComponentResult(
                status="error", note=f"support bank unusable: {exc}"[:120])
            row = _store(db, subject_id, "c3_clinical_nlp", result)
            db.commit()
            return {"subject_id": subject_id, "reading_id": row.id,
                    "status": "error", "score": None, "note": result.note}

    result = mc.call_c3(req.note_text, req.note_type,
                        req.anxiety_support, req.control_support,
                        support_set=support_set,
                        support_set_version=support_set_version,
                        note_date=req.note_date,
                        visit_count=req.visit_count,
                        subject_external_id=_external_id(db, subject_id, "c3_clinical_nlp"))'''
    if old_call in content:
        content = content.replace(old_call, new_call)
        changed = True
        print("[main.py] Patched ingest_clinical_note to use the support bank")
    elif "select_support_set(" in content:
        print("[main.py] ingest_clinical_note already patched — skipping")
    else:
        print("[main.py] WARNING: call_c3 block in ingest_clinical_note not matched — run:")
        print('           grep -n "call_c3" main.py   and patch that block by hand')

    if changed:
        open(path, "w", encoding="utf-8").write(content)
        print("[main.py] written")

def patch_modality_clients():
    path = "modality_clients.py"
    content = open(path, encoding="utf-8").read()
    changed = False

    old_sig = '''            support_set: Optional[list] = None,
            note_date: Optional[str] = None,'''
    new_sig = '''            support_set: Optional[list] = None,
            support_set_version: Optional[str] = None,
            note_date: Optional[str] = None,'''
    if old_sig in content and "support_set_version" not in content.split("def call_c3")[1].split("\n\n")[0]:
        content = content.replace(old_sig, new_sig, 1)
        changed = True
        print("[modality_clients.py] Added support_set_version param")
    elif "support_set_version" in content:
        print("[modality_clients.py] support_set_version already present")
    else:
        print("[modality_clients.py] WARNING: call_c3 signature not matched — add support_set_version param by hand")

    old_body = '''        if subject_external_id:
            request_body["subject_id"] = subject_external_id
        used_default_support = False'''
    new_body = '''        if support_set_version:
            request_body["support_set_version"] = support_set_version
        if subject_external_id:
            request_body["subject_id"] = subject_external_id
        used_default_support = False'''
    if old_body in content:
        content = content.replace(old_body, new_body)
        changed = True
        print("[modality_clients.py] Added support_set_version passthrough")
    elif 'request_body["support_set_version"]' in content:
        print("[modality_clients.py] passthrough already present")
    else:
        print("[modality_clients.py] WARNING: request_body block not matched — add passthrough by hand")

    if changed:
        open(path, "w", encoding="utf-8").write(content)
        print("[modality_clients.py] written")

def patch_test_backend():
    path = "test_backend.py"
    try:
        content = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return
    old = '''def stub_c3(note_text, note_type="progress", anxiety_support=None,
            control_support=None, support_set=None, note_date=None,
            visit_count=None, subject_external_id=None, client=None):'''
    new = '''def stub_c3(note_text, note_type="progress", anxiety_support=None,
            control_support=None, support_set=None, support_set_version=None,
            note_date=None, visit_count=None, subject_external_id=None, client=None):'''
    if old in content:
        open(path, "w", encoding="utf-8").write(content.replace(old, new))
        print("[test_backend.py] stub_c3 updated")
    elif "support_set_version" in content:
        print("[test_backend.py] stub_c3 already OK")
    else:
        print("[test_backend.py] WARNING: stub_c3 signature not matched — add support_set_version param by hand")

if __name__ == "__main__":
    patch_db_models(); print()
    patch_main(); print()
    patch_modality_clients(); print()
    patch_test_backend(); print()
    print("Done. Fix any WARNING lines above before continuing.")
