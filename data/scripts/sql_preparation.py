"""Historical SQL preparation helpers; offline adaptation of prepare_data.py.

Filtering, normalization and splitting functions are retained verbatim.
Network acquisition and historical command transport are deliberately absent.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import urllib.parse
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "raw"

PREPARED = ROOT / "prepared"

MANIFEST = ROOT / "source_manifest.json"

SEED = "remove-or-reset-v1-seed-20261011"

FRACTIONS = (0.8, 0.1, 0.1)

MAX_ARCHIVE_MEMBER = 1_000_000_000

MAX_ARCHIVE_TOTAL = 5_000_000_000

QUERY_SECONDS = 2.0

DIALECTS = ("sqlite", "mysql", "postgres", "tsql", "bigquery")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def safe_extract_spider(archive: Path) -> dict:
    target = RAW / "spider_official"
    if target.exists():
        return {
            "path": str(target.relative_to(ROOT)),
            "reused": True,
            "files": sum(1 for p in target.rglob("*") if p.is_file()),
        }
    staging = RAW / "spider_official.partial"
    if staging.exists():
        raise RuntimeError(f"partial extraction exists; inspect/remove explicitly: {staging}")
    staging.mkdir(parents=True)
    extracted = 0
    total = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename.replace("\\", "/").lstrip("/")
                normalized = re.sub(r"^spider_data/", "spider/", name)
                wanted = normalized == "spider/tables.json" or bool(
                    re.fullmatch(r"spider/database/[^/]+/[^/]+\.sqlite", normalized)
                )
                if not wanted or info.is_dir():
                    continue
                if ".." in Path(name).parts or info.file_size > MAX_ARCHIVE_MEMBER:
                    raise RuntimeError(f"unsafe archive member: {info.filename}")
                total += info.file_size
                if total > MAX_ARCHIVE_TOTAL:
                    raise RuntimeError("archive extraction size limit exceeded")
                out = (staging / normalized).resolve()
                if staging.resolve() not in out.parents:
                    raise RuntimeError(f"path traversal in archive: {info.filename}")
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                extracted += 1
        if not (staging / "spider" / "tables.json").exists() or extracted < 2:
            raise RuntimeError("Spider archive did not contain expected tables.json/databases")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(target.relative_to(ROOT)), "reused": False, "files": extracted}


DENY_ALWAYS = {
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_PRAGMA,
}

DENY_READONLY = DENY_ALWAYS | {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_TRANSACTION,
}


def authorizer(readonly: bool):
    denied = DENY_READONLY if readonly else DENY_ALWAYS

    def check(action, _a, _b, _db, _trigger):
        return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

    return check


def execute_bounded(conn: sqlite3.Connection, sql: str) -> tuple[str, str | None]:
    deadline = time.monotonic() + QUERY_SECONDS
    conn.set_authorizer(authorizer(True))
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        row = conn.execute(sql).fetchone()
        return ("empty" if row is None else "nonempty"), None
    except Exception as exc:
        return "failed", f"{type(exc).__name__}: {exc}"
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)


def infer_query(sql: str) -> tuple[exp.Expression, str, str]:
    errors = []
    for dialect in DIALECTS:
        try:
            trees = sqlglot.parse(sql, read=dialect)
            if len(trees) != 1 or trees[0] is None or not isinstance(trees[0], exp.Query):
                raise ValueError("not exactly one SELECT/CTE/set-operation query")
            tree = trees[0]
            return tree, dialect, tree.sql(dialect="sqlite")
        except Exception as exc:
            errors.append(f"{dialect}:{type(exc).__name__}")
    raise ValueError(
        "no supported dialect parsed a single read-only query (" + ",".join(errors) + ")"
    )


def context_for_dialect(context: str, dialect: str) -> list[exp.Expression]:
    trees = [x for x in sqlglot.parse(context, read=dialect) if x is not None]
    if not trees:
        raise ValueError("empty schema context")
    safe = []
    for tree in trees:
        if isinstance(tree, exp.Create):
            if str(tree.args.get("kind", "")).upper() != "TABLE":
                raise ValueError("context CREATE is not CREATE TABLE")
            if tree.find(exp.Select):
                raise ValueError("CREATE TABLE AS SELECT is not allowed")
        elif isinstance(tree, exp.Insert):
            values = tree.args.get("expression")
            if not isinstance(values, exp.Values):
                raise ValueError("only INSERT ... VALUES fixtures are allowed")
            for node in values.walk():
                if isinstance(node, (exp.Func, exp.Subquery, exp.Select, exp.Binary)):
                    raise ValueError("fixture INSERT contains executable expression")
        else:
            raise ValueError(f"unsupported context statement {type(tree).__name__}")
        safe.append(tree)
    return safe


def canonical_schema(conn: sqlite3.Connection) -> tuple[str, str]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    if not rows:
        raise ValueError("schema has no user tables")
    statements = []
    for _name, sql in rows:
        if not sql:
            continue
        try:
            normalized = sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite", pretty=False)
        except Exception:
            normalized = " ".join(sql.split())
        statements.append(normalized.rstrip(";") + ";")
    schema = "\n".join(statements)
    return schema, stable_hash(schema)


def structural_schema_hash(conn: sqlite3.Connection) -> str:
    shapes = []
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table,) in tables:
        quoted = table.replace('"', '""')
        cols = conn.execute(f'PRAGMA table_info("{quoted}")').fetchall()
        fks = conn.execute(f'PRAGMA foreign_key_list("{quoted}")').fetchall()
        shapes.append(
            {
                "columns": sorted((str(c[2]).upper(), int(c[3]), int(c[5])) for c in cols),
                "foreign_keys": sorted((str(f[5]).upper(), str(f[6]).upper()) for f in fks),
            }
        )
    shapes.sort(key=canonical_json)
    return stable_hash(canonical_json(shapes))


def sql_fingerprint(tree: exp.Expression) -> str:
    copy = tree.copy()
    for literal in list(copy.find_all(exp.Literal)):
        literal.replace(exp.Placeholder())
    return stable_hash(copy.sql(dialect="sqlite", normalize=True))


def make_record(
    source: str,
    original_split: str,
    index: int,
    db_id: str,
    question: str,
    schema: str,
    schema_id: str,
    structural_id: str,
    sql: str,
    dialect: str,
    tree: exp.Expression,
    execution: str,
    execution_error: str | None,
) -> dict:
    rid = f"{source}:{original_split}:{index:06d}"
    return {
        "id": rid,
        "source": source,
        "db_id": db_id,
        "schema_id": schema_id,
        "question": question.strip(),
        "schema": schema,
        "sql": sql.strip(),
        "dialect": dialect,
        "input": f"Schema:\n{schema}\n\nQuestion: {question.strip()}\nSQL:",
        "target": sql.strip(),
        "metadata": {
            "original_split": original_split,
            "execution_status": execution,
            "execution_error": execution_error,
            "sql_fingerprint": sql_fingerprint(tree),
            "structural_schema_id": structural_id,
        },
    }


def prepare_gretel(rejections: list[dict], counts: collections.Counter) -> list[dict]:
    records = []
    schema_cache: dict[tuple[str, str], tuple[str, str, str, bytes]] = {}
    for split in ("train", "test"):
        table = pq.read_table(RAW / f"gretel-{split}.parquet")
        for index, row in enumerate(table.to_pylist()):
            if index % 2000 == 0:
                print(f"gretel {split}: {index}/{table.num_rows}", flush=True)
            try:
                question = str(row["sql_prompt"])
                sql_original = str(row["sql"])
                context = str(row["sql_context"])
                tree, dialect, sql_sqlite = infer_query(sql_original)
                cache_key = (stable_hash(context), dialect)
                cached = schema_cache.get(cache_key)
                if cached is None:
                    conn = sqlite3.connect(":memory:")
                    try:
                        conn.set_authorizer(authorizer(False))
                        for statement in context_for_dialect(context, dialect):
                            conn.execute(statement.sql(dialect="sqlite"))
                        conn.set_authorizer(None)
                        schema, schema_id = canonical_schema(conn)
                        structural_id = structural_schema_hash(conn)
                        serialized = conn.serialize()
                    finally:
                        conn.close()
                    cached = (schema, schema_id, structural_id, serialized)
                    schema_cache[cache_key] = cached
                schema, schema_id, structural_id, serialized = cached
                conn = sqlite3.connect(":memory:")
                try:
                    conn.deserialize(serialized)
                    execution, error = execute_bounded(conn, sql_sqlite)
                finally:
                    conn.close()
                if execution == "failed":
                    raise ValueError(f"gold execution failed after SQLite transpilation: {error}")
                domain = str(row.get("domain") or "unknown")
                record = make_record(
                    "gretel",
                    split,
                    index,
                    schema_id,
                    question,
                    schema,
                    schema_id,
                    structural_id,
                    sql_sqlite,
                    "sqlite",
                    tree,
                    execution,
                    error,
                )
                # Gretel's `domain` is a broad subject label, not a database
                # identity. Use the canonical schema identity as db_id so a
                # (source, db_id) can never be split across prepared pools.
                record["metadata"]["source_domain"] = domain
                records.append(record)
                counts[f"gretel_{split}_accepted"] += 1
                counts[f"gretel_execution_{execution}"] += 1
            except Exception as exc:
                counts[f"gretel_{split}_rejected"] += 1
                reason = str(exc)
                category = reason.split(":", 1)[0][:120]
                counts[f"reject_reason::{category}"] += 1
                rejections.append(
                    {
                        "source": "gretel",
                        "original_split": split,
                        "index": index,
                        "reason": reason,
                        "question": str(row.get("sql_prompt", ""))[:500],
                        "sql": str(row.get("sql", ""))[:2000],
                    }
                )
    return records


def sqlite_uri(path: Path) -> str:
    return (
        "file:" + urllib.parse.quote(str(path.resolve()).replace("\\", "/"), safe="/:") + "?mode=ro"
    )


def prepare_spider(rejections: list[dict], counts: collections.Counter) -> tuple[list[dict], dict]:
    db_root = RAW / "spider_official" / "spider" / "database"
    table_defs = json.loads(
        (RAW / "spider_official" / "spider" / "tables.json").read_text(encoding="utf-8")
    )
    definitions = {x["db_id"]: x for x in table_defs}
    db_cache = {}
    compatibility = {
        "missing_database": [],
        "table_mismatches": [],
        "definition_count": len(definitions),
    }
    records = []
    for split in ("train", "validation"):
        table = pq.read_table(RAW / f"spider-{split}.parquet")
        for index, row in enumerate(table.to_pylist()):
            db_id = str(row["db_id"])
            question = str(row["question"])
            sql_original = str(row["query"])
            db_path = db_root / db_id / f"{db_id}.sqlite"
            try:
                if not db_path.exists():
                    compatibility["missing_database"].append(db_id)
                    raise FileNotFoundError(f"official database missing: {db_id}")
                if db_id not in db_cache:
                    conn = sqlite3.connect(sqlite_uri(db_path), uri=True)
                    try:
                        schema, schema_id = canonical_schema(conn)
                        structural_id = structural_schema_hash(conn)
                        actual_tables = {
                            r[0]
                            for r in conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table' "
                                "AND name NOT LIKE 'sqlite_%'"
                            )
                        }
                    finally:
                        conn.close()
                    expected_tables = set(
                        definitions.get(db_id, {}).get("table_names_original", [])
                    )
                    if expected_tables and {x.lower() for x in actual_tables} != {
                        x.lower() for x in expected_tables
                    }:
                        compatibility["table_mismatches"].append(
                            {
                                "db_id": db_id,
                                "sqlite": sorted(actual_tables),
                                "tables_json": sorted(expected_tables),
                            }
                        )
                    db_cache[db_id] = (schema, schema_id, structural_id)
                schema, schema_id, structural_id = db_cache[db_id]
                tree, dialect, sql_sqlite = infer_query(sql_original)
                conn = sqlite3.connect(sqlite_uri(db_path), uri=True)
                try:
                    conn.execute("PRAGMA query_only=ON")
                    execution, error = execute_bounded(conn, sql_sqlite)
                finally:
                    conn.close()
                if split == "train" and execution == "failed":
                    raise ValueError(f"gold execution failed on official SQLite DB: {error}")
                # Final Spider validation is retained even if parsing/execution fails.
                records.append(
                    make_record(
                        "spider",
                        split,
                        index,
                        db_id,
                        question,
                        schema,
                        schema_id,
                        structural_id,
                        sql_original,
                        "sqlite",
                        tree,
                        execution,
                        error,
                    )
                )
                counts[f"spider_{split}_records"] += 1
                counts[f"spider_{split}_execution_{execution}"] += 1
            except Exception as exc:
                counts[f"spider_{split}_processing_failures"] += 1
                if split == "validation":
                    # Preserve benchmark cardinality with explicit status and available fields.
                    schema, schema_id, structural_id = db_cache.get(
                        db_id, ("", "unavailable", "unavailable")
                    )
                    records.append(
                        {
                            "id": f"spider:{split}:{index:06d}",
                            "source": "spider",
                            "db_id": db_id,
                            "schema_id": schema_id,
                            "question": question,
                            "schema": schema,
                            "sql": sql_original,
                            "dialect": "sqlite",
                            "input": f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:",
                            "target": sql_original,
                            "metadata": {
                                "original_split": split,
                                "execution_status": "processing_failed",
                                "execution_error": f"{type(exc).__name__}: {exc}",
                                "sql_fingerprint": None,
                                "structural_schema_id": structural_id,
                            },
                        }
                    )
                    counts["spider_validation_records"] += 1
                else:
                    rejections.append(
                        {
                            "source": "spider",
                            "original_split": split,
                            "index": index,
                            "db_id": db_id,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "question": question[:500],
                            "sql": sql_original[:2000],
                        }
                    )
    compatibility["hf_db_ids"] = len({r["db_id"] for r in records})
    compatibility["official_sqlite_db_count"] = len(list(db_root.glob("*/*.sqlite")))
    compatibility["missing_database"] = sorted(set(compatibility["missing_database"]))
    return records, compatibility


def assignment(schema_id: str) -> str:
    value = int(stable_hash(f"{SEED}|schema|{schema_id}")[:16], 16) / 2**64
    if value < FRACTIONS[0]:
        return "train"
    if value < FRACTIONS[0] + FRACTIONS[1]:
        return "selection"
    return "validation"


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in sorted(rows, key=lambda r: r["id"]):
            f.write(canonical_json(row) + "\n")
    return sha256_file(path)
