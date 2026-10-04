"""Check that the published scripts contain the same code as the originals.

Before publication, the comments and docstrings of the Python sources were rewritten in
English, the string literals of four or more words that cited the authors' internal
working notes were reworded, and the names of internal note files and of the authors'
notes and scratch directories were replaced by neutral ones (internal/...). One variable
named after such a directory was renamed with the neutral prefix INTERNAL_. This script
computes, for each script listed in CODE_AST_SHA256, the sha256 of its syntax tree with
those parts masked: docstrings are dropped; a plain string becomes a fixed placeholder
when it has four or more words, names a .md note, or contains one of the neutral internal/...
names; an f-string whose literal text meets the same test keeps only its
interpolated expressions, in order; a variable whose name starts with INTERNAL_ becomes
a fixed name. Everything else (all other names, calls, operators, numbers,
short strings such as keys, labels and data paths, and the expressions inside every
f-string) enters the digest, so any change to executable code fails the check.

CODE_AST_SHA256 lists the digest of each ORIGINAL script, computed before publication
with Python 3.11. The tree layout differs between Python versions, so run this with
Python 3.11.

Usage (from the repository root):  python3.11 verify_code.py
"""
import ast
import hashlib
import pathlib
import sys

PLACEHOLDER = "<text>"
NEUTRAL_PREFIX = "INTERNAL_"
NEUTRAL_NAME = "INTERNAL_NAME"


def _is_docstring_holder(node):
    return isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _is_free_text(s):
    return len(s.split()) >= 4 or ".md" in s or "internal/" in s


class _Mask(ast.NodeTransformer):
    def visit_JoinedStr(self, node):
        text = "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
        if _is_free_text(text):
            node.values = [v for v in node.values if not isinstance(v, ast.Constant)]
        return self.generic_visit(node)

    def visit_FormattedValue(self, node):
        node.value = self.visit(node.value)
        return node

    def visit_Name(self, node):
        if node.id.startswith(NEUTRAL_PREFIX):
            node.id = NEUTRAL_NAME
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and _is_free_text(node.value):
            return ast.Constant(value=PLACEHOLDER)
        return node


def code_sha256(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if _is_docstring_holder(n) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) \
                    and isinstance(b.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    tree = _Mask().visit(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def main():
    if sys.version_info[:2] != (3, 11):
        print(f"warning: digests were computed with Python 3.11; this is {sys.version.split()[0]}")
    root = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (root / "CODE_AST_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        ok = code_sha256(root / rel) == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} scripts: code identical to the original")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
