"""Bounded read-only SQLite evaluation, adapted from the study evaluator.

Single-database execution agreement is NOT official Spider scoring. We compare
complete typed rows, preserving duplicates/column order; row order matters iff
the gold query has a top-level ORDER BY. There is no query extraction/salvage.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import canonical_bytes, parse_json, redact_error

SQL_SECONDS = 2.0
PARENT_SECONDS = 5.0
ROW_LIMIT = 50000
RESULT_BYTES = 8 * 1024 * 1024
WORKER_MEMORY_BYTES = 512 * 1024 * 1024


class QueryRejected(ValueError):
    pass


class SQLiteText(bytes):
    """Keep SQLite TEXT bytes distinct from BLOB, including malformed UTF-8."""


def parse_query(sql):
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError, TokenError

    if not isinstance(sql, str) or not sql.strip() or "\x00" in sql:
        raise QueryRejected("Expected one nonempty query")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError) as exc:
        raise QueryRejected(str(exc)) from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise QueryRejected("Exactly one read-only query is required")
    tree = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command, exp.Into)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise QueryRejected("Non-read-only operation")
    if any(
        isinstance(node, exp.Anonymous)
        and node.name.casefold() in {"load_extension", "writefile", "readfile", "eval"}
        for node in tree.walk()
    ):
        raise QueryRejected("External access function")
    while isinstance(tree, (exp.Subquery, exp.Paren)):
        tree = tree.this
    return bool(tree.args.get("order"))


def typed_cell(value):
    if value is None:
        return ["null", None]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return ["real", value.hex()]
    if isinstance(value, SQLiteText):
        return ["text", value.hex()]
    if isinstance(value, str):
        return ["text", value.encode("utf-8").hex()]
    if isinstance(value, bytes):
        return ["blob", value.hex()]
    raise TypeError("Unexpected SQLite storage type")


def sql_worker(payload):
    parse_query(payload["sql"])
    if os.name != "nt":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    seconds = payload.get("query_seconds", SQL_SECONDS)
    if type(seconds) not in (int, float) or not 0 < seconds <= SQL_SECONDS:
        raise ValueError("Invalid worker deadline")
    deadline = time.monotonic() + seconds
    connection = sqlite3.connect(
        Path(payload["database"]).resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=0
    )
    connection.text_factory = SQLiteText
    connection.enable_load_extension(False)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }

    def authorizer(action, first, second, database, trigger):
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(second).casefold() in {
            "load_extension",
            "writefile",
            "readfile",
            "eval",
        }:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        cursor = connection.execute(payload["sql"])
        rows, size, text_cells, invalid_utf8 = [], 0, 0, 0
        for row in cursor:
            if time.monotonic() >= deadline:
                return {"status": "timeout", "error": "SQL deadline", "limit": "query_seconds"}
            if len(rows) >= ROW_LIMIT:
                return {
                    "status": "resource_error",
                    "error": "Complete row limit exceeded",
                    "limit": "complete_rows",
                }
            for cell in row:
                if isinstance(cell, SQLiteText):
                    text_cells += 1
                    try:
                        cell.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        invalid_utf8 += 1
            tagged = [typed_cell(cell) for cell in row]
            size += len(canonical_bytes(tagged)) + 1
            if size > RESULT_BYTES:
                return {
                    "status": "resource_error",
                    "error": "Complete result exceeds 8 MiB",
                    "limit": "serialized_bytes",
                }
            rows.append(tagged)
        result = {
            "status": "ok",
            "rows": rows,
            "columns": len(cursor.description),
            "row_count": len(rows),
            "text_cells": text_cells,
            "invalid_utf8_text_cells": invalid_utf8,
        }
        if len(canonical_bytes(result)) > RESULT_BYTES:
            return {
                "status": "resource_error",
                "error": "Serialized envelope exceeds 8 MiB",
                "limit": "serialized_bytes",
            }
        if time.monotonic() >= deadline:
            return {"status": "timeout", "error": "SQL deadline", "limit": "query_seconds"}
        return result
    except sqlite3.Error as exc:
        timed_out = time.monotonic() >= deadline
        resource_error = getattr(exc, "sqlite_errorcode", None) in {
            sqlite3.SQLITE_NOMEM,
            sqlite3.SQLITE_TOOBIG,
            sqlite3.SQLITE_FULL,
        }
        return {
            "status": "timeout"
            if timed_out
            else "resource_error"
            if resource_error
            else "execution_error",
            "error": str(exc),
            "limit": "query_seconds"
            if timed_out
            else "worker_memory_bytes"
            if resource_error
            else None,
        }
    except MemoryError:
        return {
            "status": "resource_error",
            "error": "Worker memory exhausted",
            "limit": "worker_memory_bytes",
        }
    finally:
        connection.close()


class WindowsJob:
    """Specific child's 512-MiB cap + kill-on-close; assignment failure is fatal."""

    def __init__(self, pid):
        from ctypes import wintypes as w

        pointer = ctypes.c_size_t

        class Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", w.DWORD),
                ("MinimumWorkingSetSize", pointer),
                ("MaximumWorkingSetSize", pointer),
                ("ActiveProcessLimit", w.DWORD),
                ("Affinity", pointer),
                ("PriorityClass", w.DWORD),
                ("SchedulingClass", w.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", IO),
                ("ProcessMemoryLimit", pointer),
                ("JobMemoryLimit", pointer),
                ("PeakProcessMemoryUsed", pointer),
                ("PeakJobMemoryUsed", pointer),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, result, arguments in (
            ("CreateJobObjectW", w.HANDLE, [ctypes.c_void_p, w.LPCWSTR]),
            ("SetInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]),
            ("OpenProcess", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
            ("AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE]),
            ("CloseHandle", w.BOOL, [w.HANDLE]),
        ):
            function = getattr(self.api, name)
            function.restype, function.argtypes = result, arguments
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
        info.ProcessMemoryLimit = WORKER_MEMORY_BYTES
        process = None
        try:
            if not self.api.SetInformationJobObject(
                self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process = self.api.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process or not self.api.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        except OSError:
            self.close()
            raise
        finally:
            if process:
                self.api.CloseHandle(process)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def execute_sql(database, sql, deadline=None):
    try:
        parse_query(sql)
    except QueryRejected as exc:
        return {"status": "parse_error", "error": str(exc)}
    if deadline is not None and time.monotonic() >= deadline:
        return {"status": "budget_excluded", "error": "Evaluation budget exhausted"}
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "remove_reset.sql", "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    job = None
    try:
        if os.name == "nt":
            try:
                job = WindowsJob(process.pid)
            except OSError as exc:
                process.kill()
                process.communicate()
                return {
                    "status": "resource_error",
                    "error": f"Job limit failed: {exc}",
                    "limit": "windows_job",
                }
        remaining = PARENT_SECONDS if deadline is None else deadline - time.monotonic()
        stdout, stderr = process.communicate(
            json.dumps(
                {
                    "database": str(database),
                    "sql": sql,
                    "query_seconds": min(SQL_SECONDS, max(0.001, remaining)),
                }
            ),
            timeout=max(0.001, min(PARENT_SECONDS - (time.monotonic() - started), remaining)),
        )
        if process.returncode:
            return {
                "status": "resource_error",
                "error": "Worker failed",
                "worker_exit_code": process.returncode,
                "limit": "worker_exit",
                "stderr": redact_error(stderr[:2000]),
            }
        if len(stdout.encode("utf-8")) > RESULT_BYTES:
            return {
                "status": "resource_error",
                "error": "Worker output size exceeded",
                "limit": "serialized_bytes",
            }
        result = parse_json(stdout)
        result["parent_seconds"] = time.monotonic() - started
        result["serialized_result_bytes"] = len(stdout.encode("utf-8"))
        if deadline is not None and time.monotonic() >= deadline:
            result.update(status="budget_excluded", error="Evaluation budget exhausted")
        return result
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        budget = deadline is not None and time.monotonic() >= deadline
        return {
            "status": "budget_excluded" if budget else "timeout",
            "error": "Evaluation budget exhausted" if budget else "Five-second parent timeout",
            "limit": "evaluation_budget" if budget else "parent_seconds",
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        if job is not None:
            job.close()


def result_row_key(row):
    normalized = []
    for tag, value in row:
        if tag == "integer":
            normalized.append(("numeric", int(value), 1))
        elif tag == "real":
            number = float.fromhex(value)
            if math.isfinite(number):
                numerator, denominator = number.as_integer_ratio()
                normalized.append(("numeric", numerator, denominator))
            else:
                normalized.append(("numeric_nonfinite", number.hex()))
        else:
            normalized.append((tag, value))
    return tuple(normalized)


def canonical_rows(result, ordered):
    rows = [result_row_key(row) for row in result["rows"]]
    return tuple(rows if ordered else sorted(rows, key=canonical_bytes))


def results_equal(prediction, gold, ordered):
    return (
        prediction["status"] == gold["status"] == "ok"
        and prediction["columns"] == gold["columns"]
        and canonical_rows(prediction, ordered) == canonical_rows(gold, ordered)
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal SQL worker only")
    try:
        payload = parse_json(sys.stdin.read())
        result = sql_worker(payload)
    except Exception as exc:
        result = {"status": "execution_error", "error": redact_error(exc)}
    print(canonical_bytes(result).decode("utf-8"))
