"""Synchronize release notes for Git repositories under ./code.

Uses the GitHub Releases API when the repository remote points to GitHub.
Existing release documents are never removed. API failures are recorded with a
retry delay and do not stop the normal local indexer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
CODE_DIR = BASE / "code"
RELEASES_DIR = BASE / "documents" / "releases"
STATE_PATH = BASE / "index" / "release_sync_state.json"
DEFAULT_RELEASE_COUNT = 30
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
ERROR_RETRY_SECONDS = 6 * 60 * 60
REQUEST_TIMEOUT = 20
USER_AGENT = "SERGDATA-Knowledge-ReleaseSync/1.0"


def utc_now() -> float:
    return time.time()


def iso_utc(timestamp: float | None) -> str:
    if not timestamp:
        return "unknown"
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_state() -> dict:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"repositories": {}}
    except (OSError, json.JSONDecodeError):
        return {"repositories": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def git_output(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def discover_repositories() -> list[dict[str, str]]:
    found = []
    if not CODE_DIR.is_dir():
        return found
    for folder in sorted(CODE_DIR.iterdir(), key=lambda p: p.name.casefold()):
        if not folder.is_dir():
            continue
        remote = git_output(folder, "remote", "get-url", "origin")
        if not remote:
            continue
        found.append({"name": folder.name, "path": str(folder), "remote": remote})
    return found


def github_slug(remote: str) -> str | None:
    value = remote.strip()
    # Support HTTPS and common SSH remotes, including git@github.com:owner/repo.git.
    match = re.match(r"^(?:https?://github\.com/|git@github\.com:|ssh://git@github\.com/)([^\s]+)$", value, re.I)
    if not match:
        return None
    slug = match.group(1).strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    parts = slug.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        return None
    return "/".join(parts)


def api_get_releases(slug: str, count: int) -> list[dict]:
    url = f"https://api.github.com/repos/{slug}/releases?per_page={count}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, list):
        raise ValueError("GitHub API вернул неожиданный формат списка релизов")
    return payload


def safe_filename(tag: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._")
    if not name:
        name = "release"
    # Preserve uniqueness if sanitizing changes a tag.
    suffix = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8]
    return f"{name[:100]}-{suffix}.md"


def release_markdown(repo_name: str, slug: str, release: dict) -> tuple[str, str]:
    tag = str(release.get("tag_name") or "unknown-tag")
    title = str(release.get("name") or tag).strip()
    body = str(release.get("body") or "").strip()
    html_url = str(release.get("html_url") or f"https://github.com/{slug}/releases/tag/{urllib.parse.quote(tag, safe='')}")
    published = str(release.get("published_at") or release.get("created_at") or "unknown")
    prerelease = bool(release.get("prerelease"))
    draft = bool(release.get("draft"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if not body:
        body = "Официальное описание релиза пустое. Не подменять его предположениями по исходному коду."
    text = (
        f"# {repo_name} — {title}\n\n"
        f"Тип документа: официальные релиз-ноты GitHub\n"
        f"Репозиторий: {slug}\n"
        f"Тег: {tag}\n"
        f"Дата публикации: {published}\n"
        f"Предварительный релиз: {'да' if prerelease else 'нет'}\n"
        f"Черновик: {'да' if draft else 'нет'}\n"
        f"Источник: {html_url}\n"
        f"SHA-256 описания: {digest}\n\n"
        f"## Описание релиза\n\n{body}\n"
    )
    return tag, text


def sync_git_tag_summaries(repo: Path, repo_name: str, remote: str, count: int) -> int:
    """Create clearly labelled local Git-history summaries when Releases API has no releases.

    These are not presented as official release notes. Only a bounded number of
    recent tags are processed, and existing summaries are skipped by content.
    """
    raw_tags = git_output(repo, "tag", "--sort=-version:refname")
    tags = [line.strip() for line in raw_tags.splitlines() if line.strip()][:count]
    if not tags:
        print(f"[{repo_name}] Нет Git-тегов для резервного источника")
        return 0

    target_dir = RELEASES_DIR / repo_name / "git-tag-summaries"
    written = 0
    for index, tag in enumerate(tags):
        previous_tag = tags[index + 1] if index + 1 < len(tags) else ""
        range_spec = f"{previous_tag}..{tag}" if previous_tag else tag
        log = git_output(
            repo,
            "log",
            "--no-merges",
            "--format=%h %cs %s",
            "-n",
            "100",
            range_spec,
        )
        if not log:
            log = "Для этого тега локальная история коммитов недоступна или пуста."
        remote_url = remote
        text = (
            f"# {repo_name} — Git tag {tag}\n\n"
            "Тип документа: сводка Git-коммитов, НЕ официальные релиз-ноты\n"
            f"Репозиторий: {remote_url}\n"
            f"Тег: {tag}\n"
            f"Сравнение с предыдущим тегом: {previous_tag or 'нет предыдущего тега в локальном списке'}\n\n"
            "## Коммиты в диапазоне\n\n"
            f"```text\n{log}\n```\n\n"
            "Примечание: сообщения коммитов не гарантируют полного описания пользовательских изменений. "
            "Использовать как вспомогательный источник, проверяя вывод по исходному коду.\n"
        )
        path = target_dir / safe_filename(tag)
        try:
            old = path.read_text(encoding="utf-8")
        except OSError:
            old = None
        if old == text:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        written += 1
    print(f"[{repo_name}] Резервная Git-история: сохранено/обновлено сводок {written} из {len(tags)} тегов")
    return written


def sync(*, count: int = DEFAULT_RELEASE_COUNT, force: bool = False, dry_run: bool = False) -> int:
    count = max(1, min(int(count), 100))
    state = load_state()
    repos_state = state.setdefault("repositories", {})
    repos = discover_repositories()
    if not repos:
        print(f"Репозитории Git не обнаружены в {CODE_DIR}")
        return 0

    print(f"Найдено Git-репозиториев: {len(repos)}; глубина истории: {count}")
    total_new = total_updated = total_skipped = total_errors = 0
    state_changed = False

    for repo in repos:
        name, remote = repo["name"], repo["remote"]
        slug = github_slug(remote)
        if not slug:
            print(f"[{name}] Пропуск релизов: источник не распознан как GitHub ({remote})")
            total_skipped += 1
            continue

        previous = repos_state.get(slug, {})
        now = utc_now()
        retry_after = float(previous.get("retry_after", 0) or 0)
        last_checked = float(previous.get("last_successful_check", 0) or 0)
        if not force and retry_after > now:
            print(f"[{name}] Пропуск проверки до {iso_utc(retry_after)} после предыдущей ошибки")
            total_skipped += 1
            continue
        if not force and last_checked and now - last_checked < CHECK_INTERVAL_SECONDS:
            print(f"[{name}] Пропуск: проверялся {iso_utc(last_checked)}, следующий плановый опрос через 24 ч")
            total_skipped += 1
            continue

        print(f"[{name}] Проверяю GitHub Releases: {slug}")
        if dry_run:
            print(f"[{name}] Предварительный просмотр: запрос списка до {count} релизов; файлы не меняются")
            continue

        try:
            releases = api_get_releases(slug, count)
        except urllib.error.HTTPError as exc:
            retry = now + ERROR_RETRY_SECONDS
            repos_state[slug] = {
                **previous,
                "remote": remote,
                "last_error": f"HTTP {exc.code}: {exc.reason}",
                "last_error_at": now,
                "retry_after": retry,
            }
            state_changed = True
            total_errors += 1
            if exc.code == 403 or exc.code == 429:
                print(f"[{name}] GitHub ограничил запросы (HTTP {exc.code}); повтор не раньше {iso_utc(retry)}")
            else:
                print(f"[{name}] Ошибка GitHub API HTTP {exc.code}; повтор не раньше {iso_utc(retry)}")
            continue
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            retry = now + ERROR_RETRY_SECONDS
            repos_state[slug] = {
                **previous,
                "remote": remote,
                "last_error": f"{type(exc).__name__}: {exc}",
                "last_error_at": now,
                "retry_after": retry,
            }
            state_changed = True
            total_errors += 1
            print(f"[{name}] Сетевая/форматная ошибка: {type(exc).__name__}: {exc}; локальные данные сохранены")
            continue

        target_dir = RELEASES_DIR / name
        if not releases:
            print(f"[{name}] GitHub API не вернул опубликованных релизов; использую явно помеченную сводку Git-тегов")
            try:
                sync_git_tag_summaries(Path(repo["path"]), name, remote, count)
            except Exception as exc:
                print(f"[{name}] Не удалось сформировать сводки Git-тегов: {type(exc).__name__}: {exc}")
        else:
            for release in releases:
                if not isinstance(release, dict) or release.get("draft"):
                    continue
                tag, content = release_markdown(name, slug, release)
                path = target_dir / safe_filename(tag)
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                try:
                    old_content = path.read_text(encoding="utf-8")
                except OSError:
                    old_content = None
                if old_content == content:
                    total_skipped += 1
                    continue
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    tmp.write_text(content, encoding="utf-8")
                    os.replace(tmp, path)
                if old_content is None:
                    total_new += 1
                    print(f"[{name}] Новый релиз: {tag}")
                else:
                    total_updated += 1
                    print(f"[{name}] Обновлено описание: {tag}")

        repos_state[slug] = {
            "remote": remote,
            "last_successful_check": now,
            "retry_after": 0,
            "last_error": None,
            "release_count_returned": len(releases),
            "requested_release_count": count,
            "tags": [str(item.get("tag_name")) for item in releases if isinstance(item, dict) and item.get("tag_name")],
        }
        state_changed = True

    if state_changed and not dry_run:
        save_state(state)
    print(
        "Синхронизация релизов завершена: "
        f"новых={total_new}, обновлено={total_updated}, пропущено={total_skipped}, ошибок={total_errors}"
    )
    print("При ошибках API существующие документы не удаляются; локальная индексация может продолжаться.")
    return 1 if total_errors and not (total_new or total_updated) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Синхронизация релизов репозиториев SERGDATA Knowledge")
    parser.add_argument("--count", type=int, default=DEFAULT_RELEASE_COUNT, help="Число последних релизов (по умолчанию 30; максимум 100)")
    parser.add_argument("--force", action="store_true", help="Проверить источники независимо от интервала/задержки")
    parser.add_argument("--dry-run", action="store_true", help="Показать план без сетевых запросов/изменения файлов")
    args = parser.parse_args()
    return sync(count=args.count, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
