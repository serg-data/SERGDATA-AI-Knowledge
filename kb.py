import argparse
import hashlib
import os
import re
import sqlite3
import zipfile
from pathlib import Path

from document_readers import SUPPORTED_EXTENSIONS, extract_text

BASE = Path(__file__).resolve().parent
DB = BASE / "index" / "knowledge.db"
SOURCES = [BASE / "documents", BASE / "code"]

EXTENSIONS = SUPPORTED_EXTENSIONS

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "out",
    ".idea",
    ".vscode",
}

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB
CHUNK_LINES = 100
OVERLAP_LINES = 15
MAX_QUERY_TERMS = 12


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)

    con.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            modified REAL NOT NULL,
            content TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS search
        USING fts5(path, content, tokenize='unicode61')
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            digest TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search
        USING fts5(path, content, tokenize='unicode61')
    """)
    return con


def excluded(path):
    return any(part in EXCLUDED_DIRS for part in path.parts)


def split_chunks(lines):
    step = max(1, CHUNK_LINES - OVERLAP_LINES)

    for start in range(0, len(lines), step):
        end = min(start + CHUNK_LINES, len(lines))
        content = "".join(lines[start:end])

        if content.strip():
            yield start + 1, end, content

        if end == len(lines):
            break


def read_text_robust(path):
    """Read a file and include the failing path in any OS error."""
    try:
        with open(
            os.fspath(path),
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as file:
            return file.read()
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"ошибка чтения ({type(exc).__name__}): {exc}",
            os.fspath(path),
        ) from exc


def index_files():
    con = connect()
    updated = 0
    skipped = 0
    unchanged = 0

    try:
        for folder in SOURCES:
            if not folder.exists():
                print(f"Источник отсутствует, пропуск: {folder}")
                continue

            try:
                for path in folder.rglob("*"):
                    try:
                        if not path.is_file():
                            continue

                        if excluded(path.relative_to(folder)):
                            continue

                        if path.suffix.lower() not in EXTENSIONS:
                            continue

                        relpath = path.relative_to(BASE).as_posix()

                        try:
                            stat = path.stat()

                            if stat.st_size > MAX_FILE_BYTES:
                                print(f"Пропущен слишком большой файл: {relpath}")
                                skipped += 1
                                continue

                            old = con.execute(
                                "SELECT id, modified FROM files WHERE path = ?",
                                (relpath,),
                            ).fetchone()

                            has_chunks = con.execute(
                                "SELECT 1 FROM chunks WHERE path = ? LIMIT 1",
                                (relpath,),
                            ).fetchone()

                            if old and old[1] == stat.st_mtime and has_chunks:
                                unchanged += 1
                                continue

                            content = extract_text(path)

                            if not content.strip():
                                print(f"Пропущен пустой файл: {relpath}")
                                skipped += 1
                                continue

                            lines = content.splitlines(keepends=True)
                            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

                            # One file's index is updated atomically.
                            with con:
                                old = con.execute(
                                    "SELECT id FROM files WHERE path = ?",
                                    (relpath,),
                                ).fetchone()

                                if old:
                                    file_id = old[0]

                                    con.execute(
                                        "DELETE FROM search WHERE path = ?",
                                        (relpath,),
                                    )
                                    con.execute(
                                        "DELETE FROM chunk_search WHERE path = ?",
                                        (relpath,),
                                    )
                                    con.execute(
                                        "DELETE FROM chunks WHERE path = ?",
                                        (relpath,),
                                    )
                                    con.execute(
                                        "UPDATE files SET modified=?, content=? "
                                        "WHERE id=?",
                                        (stat.st_mtime, content, file_id),
                                    )
                                else:
                                    cur = con.execute(
                                        "INSERT INTO files(path, modified, content) "
                                        "VALUES (?, ?, ?)",
                                        (relpath, stat.st_mtime, content),
                                    )
                                    file_id = cur.lastrowid

                                con.execute(
                                    "INSERT INTO search(rowid, path, content) "
                                    "VALUES (?, ?, ?)",
                                    (file_id, relpath, content),
                                )

                                for start_line, end_line, chunk in split_chunks(lines):
                                    cur = con.execute(
                                        "INSERT INTO chunks "
                                        "(path, start_line, end_line, content, digest) "
                                        "VALUES (?, ?, ?, ?, ?)",
                                        (
                                            relpath,
                                            start_line,
                                            end_line,
                                            chunk,
                                            digest,
                                        ),
                                    )

                                    con.execute(
                                        "INSERT INTO chunk_search"
                                        "(rowid, path, content) VALUES (?, ?, ?)",
                                        (cur.lastrowid, relpath, chunk),
                                    )

                            updated += 1

                            if updated % 100 == 0:
                                print(f"Обработано файлов: {updated}")

                        except (OSError, sqlite3.Error, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                            print(f"Пропущен {relpath}: {type(exc).__name__}: {exc}")
                            skipped += 1

                    except OSError as exc:
                        print(
                            f"Ошибка доступа к пути {path}: {type(exc).__name__}: {exc}"
                        )
                        skipped += 1

            except OSError as exc:
                print(f"Ошибка обхода папки {folder}: {type(exc).__name__}: {exc}")
                skipped += 1

        con.commit()

        print(
            f"Индексация завершена. Обновлено файлов: {updated}; "
            f"без изменений: {unchanged}; пропущено: {skipped}"
        )
        print(
            "Всего фрагментов:",
            con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        )

    finally:
        con.close()


def _query_terms(query):
    terms = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
    unique = []
    seen = set()

    for term in terms:
        term = term.strip(".-_")
        folded = term.casefold()

        if term and folded not in seen:
            unique.append(term)
            seen.add(folded)

    return unique[:MAX_QUERY_TERMS]


def _fts_quote(term):
    return '"' + term.replace('"', '""') + '"'


def _run_chunk_query(con, expression, limit):
    return con.execute(
        """
        SELECT
            chunks.path,
            chunks.start_line,
            chunks.end_line,
            snippet(chunk_search, 1, '>>>', '<<<', ' ... ', 24),
            bm25(chunk_search, 4.0, 1.0) AS score
        FROM chunk_search
        JOIN chunks ON chunks.id = chunk_search.rowid
        WHERE chunk_search MATCH ?
        ORDER BY score
        LIMIT ?
    """,
        (expression, limit),
    ).fetchall()


def search_files(query, limit=5):
    """Search using strict, broad and per-term routes; merge without duplicates."""
    terms = _query_terms(query)

    if not terms:
        print("Укажи поисковый запрос.")
        return

    safe_limit = max(1, min(limit, 20))
    candidate_limit = min(100, max(30, safe_limit * 10))
    con = connect()
    ranked = {}

    try:
        routes = []
        if len(terms) > 1:
            routes.append(("strict", " AND ".join(_fts_quote(term) for term in terms)))
        routes.append(("broad", " OR ".join(_fts_quote(term) for term in terms)))
        # Single-term routes help retrieve relevant material when one query word
        # is rare, misspelled, or absent from an otherwise useful chunk.
        if len(terms) > 1:
            routes.extend(
                (f"term:{i}", _fts_quote(term)) for i, term in enumerate(terms[:6])
            )

        for route_name, expression in routes:
            try:
                candidates = _run_chunk_query(con, expression, candidate_limit)
            except sqlite3.OperationalError:
                # A malformed/unsupported route must not suppress other routes.
                continue
            for rank, row in enumerate(candidates, start=1):
                key = (row[0], row[1], row[2])
                state = ranked.setdefault(
                    key, {"row": row, "rrf": 0.0, "strict": False}
                )
                state["rrf"] += 1.0 / (60 + rank)
                if route_name == "strict":
                    state["strict"] = True
                # Prefer the better FTS score if the same chunk is returned twice.
                if row[4] < state["row"][4]:
                    state["row"] = row

        def ranking_key(state):
            row = state["row"]
            path, _start, _end, snippet_text, score = row
            searchable = f"{path} {snippet_text}".casefold()
            coverage = sum(1 for term in terms if term.casefold() in searchable)
            # Full-query matches lead; then prefer term coverage, multi-route
            # consensus, and FTS relevance. Negative BM25 values are better.
            return (not state["strict"], -coverage, -state["rrf"], score)

        rows = [
            state["row"]
            for state in sorted(ranked.values(), key=ranking_key)[:safe_limit]
        ]

    except sqlite3.OperationalError as exc:
        print(f"Ошибка поиска: {exc}")
        return
    finally:
        con.close()

    if not rows:
        print(
            "Совпадений нет. Попробуй более короткий запрос или отдельный технический термин."
        )
        return

    for path, start, end, snippet_text, _score in rows:
        print(f"\n--- {path}:{start}-{end} ---\n{snippet_text}")


def main():
    parser = argparse.ArgumentParser(description="Локальная база знаний SERGDATA")
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = sub.add_parser(
        "index",
        help="Синхронизировать релизы и индексировать документы и код",
    )
    index_parser.add_argument(
        "--no-release-sync",
        action="store_true",
        help="Не обращаться к сетевым источникам релизов; индексировать только локальные файлы",
    )
    index_parser.add_argument(
        "--release-count",
        type=int,
        default=30,
        help="Сколько последних релизов запрашивать при синхронизации (по умолчанию 30)",
    )

    search = sub.add_parser(
        "search",
        help="Искать по базе",
    )
    search.add_argument(
        "query",
        nargs="+",
        help="Слова для поиска",
    )
    search.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.command == "index":
        if not args.no_release_sync:
            try:
                from release_sync import sync as sync_releases
                sync_releases(count=args.release_count)
            except Exception as exc:
                # Release synchronization must never prevent local indexing.
                print(f"Синхронизация релизов не удалась; продолжаю локальную индексацию: {type(exc).__name__}: {exc}")
        else:
            print("Синхронизация релизов отключена параметром --no-release-sync")
        index_files()
    elif args.command == "search":
        search_files(" ".join(args.query), args.limit)


if __name__ == "__main__":
    main()
