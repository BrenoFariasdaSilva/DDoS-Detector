# comments_standardizer.py
# Standardizes Python comments in .py files located in the root directory (non-recursive),
# including inline comments (code + # comment). Avoids modifying strings and shebangs.

import os
import tokenize
from io import BytesIO

ROOT_DIR = r"D:\Backup\GitHub\Public\DDoS-Detector"


def standardize_comment(raw: str) -> str:
    """
    Receives a full comment token starting with '#'.
    Ensures exactly one space after '#' and capitalizes the first letter of the text.
    """
    body = raw[1:].strip()  # remove '#' and trim spaces
    if not body:
        return "#"
    return "# " + body[0].upper() + body[1:]


def process_file(file_path: str) -> None:
    with open(file_path, "rb") as f:
        source = f.read()

    tokens = list(tokenize.tokenize(BytesIO(source).readline))
    modified = False
    new_tokens = []

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            new_comment = standardize_comment(tok.string)
            if new_comment != tok.string:
                tok = tokenize.TokenInfo(
                    tok.type,
                    new_comment,
                    tok.start,
                    tok.end,
                    tok.line,
                )
                modified = True
        new_tokens.append(tok)

    if modified:
        new_source = tokenize.untokenize(new_tokens)
        with open(file_path, "wb") as f:
            f.write(new_source)


def main():
    for filename in os.listdir(ROOT_DIR):
        if filename.endswith(".py"):
            file_path = os.path.join(ROOT_DIR, filename)
            if os.path.isfile(file_path):
                process_file(file_path)


if __name__ == "__main__":
    main()
