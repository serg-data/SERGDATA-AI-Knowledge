import re
import sqlite3
import zipfile
from pathlib import Path

from document_readers import SUPPORTED_EXTENSIONS, extract_text, extract_pdf_page_range

from mcp.server.fastmcp import FastMCP

BASE = Path(__file__).resolve().parent
DB = BASE / "index" / "knowledge.db"

ALLOWED_DIRS = (
    BASE / "documents",
    BASE / "code",
)

MAX_FILE_CHARS = 20000
MAX_QUERY_TERMS = 12
MAX_FALLBACK_CANDIDATES = 100
MAX_QUERY_CHARS = 1000
RRF_K = 60

mcp = FastMCP("SERGDATA Knowledge Base")


def _query_terms(query: str) -> list[str]:
    """Extract unique search terms and limit query complexity."""
    terms = re.findall(r"[\w.-]+", query, flags=re.UNICODE)

    unique_terms = []
    seen = set()

    for term in terms:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)

        if len(unique_terms) >= MAX_QUERY_TERMS:
            break

    return unique_terms


def _fts_expression(terms: list[str], operator: str = "AND") -> str:
    """Build a safely quoted FTS5 expression."""
    quoted = ['"' + term.replace('"', '""') + '"' for term in terms]
    return f" {operator} ".join(quoted)


def _search_rows(
    con: sqlite3.Connection,
    expression: str,
    limit: int,
) -> list[tuple]:
    """Search indexed chunks using SQLite FTS5."""
    return con.execute(
        """
        SELECT
            chunks.path,
            chunks.start_line,
            chunks.end_line,
            snippet(
                chunk_search,
                1,
                '>>>',
                '<<<',
                ' ... ',
                35
            ),
            bm25(chunk_search, 4.0, 1.0)
        FROM chunk_search
        JOIN chunks
            ON chunks.id = chunk_search.rowid
        WHERE chunk_search MATCH ?
        ORDER BY bm25(chunk_search)
        LIMIT ?
        """,
        (expression, limit),
    ).fetchall()


def _rank_fallback_rows(
    rows: list[tuple],
    terms: list[str],
) -> list[tuple]:
    """Prioritize fallback results with more query terms in path/snippet."""
    ranked = []

    for row in rows:
        path, start_line, end_line, snippet, bm25_score = row
        searchable_text = f"{path} {snippet}".casefold()

        matched_terms = sum(1 for term in terms if term.casefold() in searchable_text)

        ranked.append(
            (
                matched_terms,
                bm25_score,
                row,
            )
        )

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked]


@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> str:
    """Search technical knowledge with strict, broad and identifier-term routes.

    Results are deduplicated and merged across routes. Exact all-term matches
    are prioritized, while broad/single-term routes improve recall.
    """
    if not isinstance(query, str):
        return "Поисковый запрос должен быть строкой."
    query = query.strip()
    if not query:
        return "Укажи поисковый запрос."
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]

    terms = _query_terms(query)
    if not terms:
        return "Укажи поисковый запрос."

    try:
        limit = int(limit)
    except (TypeError, ValueError, OverflowError):
        return "Параметр limit должен быть целым числом."
    limit = max(1, min(limit, 10))

    if not DB.exists():
        return "База знаний не найдена. Сначала выполни индексацию."

    candidate_limit = min(MAX_FALLBACK_CANDIDATES, max(30, limit * 10))
    merged = {}
    routes = []
    if len(terms) > 1:
        routes.append(("strict", _fts_expression(terms, "AND")))
    routes.append(("broad", _fts_expression(terms, "OR")))
    # Keep the extra work bounded: single-term retrieval is most useful for
    # technical identifiers and cases where one term is absent from a chunk.
    if len(terms) > 1:
        routes.extend(
            (f"term:{i}", _fts_expression([term])) for i, term in enumerate(terms[:6])
        )

    try:
        with sqlite3.connect(DB) as con:
            for route_name, expression in routes:
                try:
                    rows = _search_rows(con, expression, candidate_limit)
                except sqlite3.OperationalError:
                    # One route failing should not break the remaining search.
                    continue
                for rank, row in enumerate(rows, start=1):
                    path, start_line, end_line, snippet, score = row
                    key = (path, start_line, end_line)
                    state = merged.setdefault(
                        key,
                        {"row": row, "rrf": 0.0, "strict": False, "routes": 0},
                    )
                    state["rrf"] += 1.0 / (RRF_K + rank)
                    state["routes"] += 1
                    if route_name == "strict":
                        state["strict"] = True
                    if score < state["row"][4]:
                        state["row"] = row
    except sqlite3.Error as exc:
        return f"Ошибка поиска в базе знаний: {exc}"

    def ranking_key(state):
        row = state["row"]
        path, _start, _end, snippet, score = row
        searchable = f"{path} {snippet}".casefold()
        coverage = sum(1 for term in terms if term.casefold() in searchable)
        return (not state["strict"], -coverage, -state["rrf"], score)

    ordered = sorted(merged.values(), key=ranking_key)[:limit]
    if not ordered:
        return (
            "Совпадений не найдено. Попробуй более короткий запрос с названиями "
            "файлов, функций, версий или техническими терминами."
        )

    results = []
    for state in ordered:
        path, start_line, end_line, snippet, _score = state["row"]
        modes = "точное совпадение" if state["strict"] else "расширенный поиск"
        results.append(
            f"Режим поиска: {modes}; совпадений маршрутов: {state['routes']}\n"
            f"Файл: {path}:{start_line}-{end_line}\n"
            f"Найденный фрагмент:\n{snippet}"
        )
    return "\n\n---\n\n".join(results)


@mcp.tool()
def read_knowledge_file(
    path: str,
    start_line: int = 1,
    end_line: int = 0,
) -> str:
    """Read a file or a specific line range from the knowledge base.

    Use the relative path returned by search_knowledge.
    If end_line is 0, read from start_line to the end of the file.
    """

    try:
        start_line = int(start_line)
        end_line = int(end_line)
    except (TypeError, ValueError, OverflowError):
        return "start_line и end_line должны быть целыми числами."

    if start_line < 1:
        return "start_line должен быть не меньше 1."

    if end_line < 0:
        return "end_line должен быть равен 0 или быть положительным числом."

    if end_line != 0 and end_line < start_line:
        return "end_line не может быть меньше start_line."

    candidate = (BASE / path).resolve()

    allowed = any(
        candidate != folder.resolve() and folder.resolve() in candidate.parents
        for folder in ALLOWED_DIRS
    )

    if not allowed or not candidate.is_file():
        return "Файл не найден или путь находится вне разрешённых папок."

    allowed_extensions = SUPPORTED_EXTENSIONS

    if candidate.suffix.lower() not in allowed_extensions:
        return "Этот тип файла не разрешён для чтения."

    try:
        if candidate.suffix.lower() == ".pdf":
            if start_line > 1_000_000 or end_line > 1_000_000:
                return "Номер страницы слишком велик."
            text = extract_pdf_page_range(candidate, start_line, end_line)
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + "\n[Фрагмент обрезан: достигнут лимит символов.]"
            selected_lines = text.splitlines(keepends=True)
            label = f"страницы {start_line}-{end_line if end_line else 'конец'}"
        else:
            text = extract_text(candidate)
            lines = text.splitlines(keepends=True)
            if start_line > len(lines):
                lines = []
            else:
                stop = end_line if end_line else len(lines)
                lines = lines[start_line - 1:stop]
            selected_lines = []
            total_chars = 0
            for line in lines:
                if total_chars + len(line) > MAX_FILE_CHARS:
                    selected_lines.append("\n[Фрагмент обрезан: достигнут лимит символов.]\n")
                    break
                selected_lines.append(line)
                total_chars += len(line)
            label = f"строки {start_line}-{end_line if end_line else 'конец'}"
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        return f"Не удалось прочитать файл: {exc}"

    if not selected_lines:
        return "В указанном диапазоне строк содержимое не найдено."

    relative_path = candidate.relative_to(BASE)

    return (
        f"Файл: {relative_path}\n"
        f"Запрошенный диапазон: {label}\n\n" + "".join(selected_lines)
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
