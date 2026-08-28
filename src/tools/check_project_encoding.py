from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

TEXT_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".txt", ".md", ".toml", ".ini", ".cfg", ".csv"
}

# Các dấu hiệu điển hình của UTF-8 đã bị giải mã nhầm theo CP1252/Latin-1.
MOJIBAKE_MARKERS = (
    "Ã", "Â", "Ä", "áº", "á»", "Æ", "Å", "ðŸ", "â€", "â€™", "â€œ", "â€", "â€“", "â€”"
)

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "ENV", "__pycache__",
    ".idea", ".vscode", "data", "submission", "submissions"
}


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def try_repair_cp1252_utf8(text: str) -> str | None:
    """
    Thử sửa dạng mojibake phổ biến:
        'mÃºa lÃ¢n' -> 'múa lân'
        'Ä‘'         -> 'đ'
        'ðŸŽ¬'       -> emoji đúng

    Chỉ trả bản sửa nếu:
    - encode CP1252 + decode UTF-8 thành công;
    - số dấu hiệu mojibake giảm.
    """
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None

    if mojibake_score(repaired) < mojibake_score(text):
        return repaired
    return None


def inspect_file(path: Path) -> dict:
    raw = path.read_bytes()

    try:
        text = raw.decode("utf-8")
        utf8_ok = True
        decode_error = None
    except UnicodeDecodeError as exc:
        return {
            "path": str(path),
            "utf8_ok": False,
            "decode_error": str(exc),
            "mojibake_lines": [],
        }

    bad_lines = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        score = mojibake_score(line)

        if score <= 0:
            continue

        repaired = try_repair_cp1252_utf8(line)

        # Một số ký tự như "Ã", "Â" có thể là Unicode tiếng Việt hợp lệ.
        # Chỉ coi là mojibake khi phép CP1252 -> UTF-8 thực sự sửa được
        # và try_repair_cp1252_utf8() đã xác nhận mojibake_score giảm.
        if repaired is None:
            continue

        bad_lines.append({
            "line": line_no,
            "score": score,
            "text": line,
            "suggested": repaired,
        })

    return {
        "path": str(path),
        "utf8_ok": utf8_ok,
        "decode_error": decode_error,
        "mojibake_lines": bad_lines,
    }


def repair_file_safely(path: Path) -> tuple[bool, int]:
    """
    Chỉ sửa từng dòng khi phép chuyển đổi làm GIẢM rõ ràng dấu hiệu mojibake.
    Tạo .bak trước khi ghi.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = 0
    new_lines = []

    for line in lines:
        body = line.rstrip("\r\n")
        ending = line[len(body):]

        if mojibake_score(body) > 0:
            repaired = try_repair_cp1252_utf8(body)
            if repaired is not None:
                body = repaired
                changed += 1

        new_lines.append(body + ending)

    if changed == 0:
        return False, 0

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("".join(new_lines))
    return True, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm tra UTF-8 và mojibake trong source AIC 2026."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root. Mặc định: thư mục hiện tại."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Sửa an toàn các dòng mojibake nhận diện chắc chắn và tạo file .bak."
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = list(iter_text_files(root))

    utf8_errors = 0
    suspicious_files = 0
    suspicious_lines = 0

    print("=" * 72)
    print("AIC 2026 - SOURCE ENCODING AUDIT")
    print(f"Root: {root}")
    print(f"Text files scanned: {len(files)}")
    print("=" * 72)

    for path in files:
        info = inspect_file(path)

        if not info["utf8_ok"]:
            utf8_errors += 1
            print(f"[UTF8 ERROR] {path.relative_to(root)}")
            print(f"  {info['decode_error']}")
            continue

        bad = info["mojibake_lines"]
        if not bad:
            continue

        suspicious_files += 1
        suspicious_lines += len(bad)
        print(f"\n[MOJIBAKE?] {path.relative_to(root)} - {len(bad)} line(s)")

        for item in bad[:8]:
            print(f"  L{item['line']}: {item['text'][:180]}")
            if item["suggested"] is not None:
                print(f"        -> {item['suggested'][:180]}")

        if len(bad) > 8:
            print(f"  ... còn {len(bad) - 8} dòng.")

        if args.fix:
            changed, count = repair_file_safely(path)
            if changed:
                print(f"  [FIXED] {count} dòng; backup: {path.name}.bak")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"UTF-8 decode errors : {utf8_errors}")
    print(f"Mojibake files      : {suspicious_files}")
    print(f"Mojibake lines      : {suspicious_lines}")
    print("=" * 72)

    if utf8_errors:
        raise SystemExit(2)
    if suspicious_files and not args.fix:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
