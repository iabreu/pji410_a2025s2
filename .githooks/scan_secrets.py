import re
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

PATTERNS = {
    "generic_password": re.compile(r"(?i)(password|senha)\s*[:=]\s*['\"]?[^'\"\n]{4,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(
        r"(?i)aws(.{0,20})?(secret|access).{0,5}['\"][0-9A-Za-z/+]{30,}['\"]"
    ),
    "private_key_block": re.compile(
        r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"
    ),
    "token_like": re.compile(
        r"(?i)(token|api_key|apikey|bearer)\s*[:=]\s*['\"][A-Za-z0-9-_]{10,}['\"]"
    ),
    "hex_entropy": re.compile(r"(?<![A-F0-9])[A-F0-9]{32}(?![A-F0-9])"),
    # Specific: flag zip_password assignments ONLY when value is a non-empty literal (>=3 chars) not blank
    # Matches: zip_password = "abc123"  (flags)
    # Skips:   zip_password = ""       (no flag)
    # Skips:   zip_password = os.getenv("ZIP_PASS") (not a simple quoted literal)
    "zip_password_literal": re.compile(r"zip_password\s*=\s*['\"](?P<val>[^'\"]{3,})['\"]"),
}

BINARY_EXT = {"png", "jpg", "jpeg", "gif", "pdf", "gz", "tar", "jar", "mp4", "mp3"}


def is_binary(path: Path) -> bool:
    if path.suffix.lower().lstrip(".") in BINARY_EXT:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return True
    except Exception:
        return True
    return False


def scan_file(path: Path) -> list[tuple[str, str]]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [("read_error", f"{path}: {e}")]
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            snippet = match.group(0)[:120]
            findings.append((name, snippet))
    return findings


def main(paths: list[str]) -> int:
    any_findings = False
    for p in paths:
        fp = Path(p)
        if not fp.exists() or fp.is_dir():
            continue
        if is_binary(fp):
            continue
        results = scan_file(fp)
        if results:
            any_findings = True
            print(f"[POTENTIAL SECRET] {fp}")
            for kind, snippet in results:
                print(f"  - {kind}: {snippet}")
    if any_findings:
        logging.error(
            "\nCommit abortado: remova ou mascare os segredos acima (ou confirme falso positivo)."
        )
        logging.info("Para ignorar (não recomendado): git commit --no-verify")
        return 1
    return 0


if __name__ == "__main__":
    # Expect list of staged files via args
    sys.exit(main(sys.argv[1:]))
