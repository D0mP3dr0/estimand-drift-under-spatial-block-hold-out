"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
Field-by-field comparison of a re-run of the revision-1 scripts with the stamped records in results/fase6.

Usage:  python analysis/revision_r1/compare_fase6.py <rerun_dir> [reference_dir] [file ...]
reference_dir defaults to results/fase6. Numeric leaves are compared exactly (largest absolute difference is
reported); string leaves are listed by path when they differ (provenance fields such as script digest,
command, timestamp and the digests of inputs are expected to differ). Exit status 1 if any numeric leaf,
boolean, null or key set differs.
"""
import json
import math
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_FILES = ["V1_resumo.json", "V1_por_sorteio.json", "V2_resumo.json", "V2bis_resumo.json",
                 "V3_tabela_celulas.json", "V4_tabela_vies.json", "E6_resumo.json", "E6_por_sorteio.json"]
TEXT_SKIP = {"data_utc", "tempo_s", "tempo_total_s"}  # wall-clock fields, not results
# Provenance fields: file names, digests of scripts and inputs, command lines. They differ by construction
# (the scripts and inputs of this repository are the cleaned copies); they are counted, not compared (numbers are always compared).
PROV = re.compile(r"sha256|fonte|comando|venv|script|criterio|caminho|chave_kfold|item_do_parecer|artefato_localizado|"
                  r"nota|copia|parciais|referencias_lidas|rotina_.*reusada|origem")


def walk(a, b, path, acc):
    prov = any(PROV.search(c.split("[")[0]) for c in path.split(".") if c)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diff = sorted(set(a) ^ set(b))
            if prov or all(PROV.search(k) for k in diff):
                acc["prov"] += 1
            else:
                acc["keys"].append((path, diff))
        for k in a.keys() & b.keys():
            walk(a[k], b[k], f"{path}.{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["keys"].append((path, f"len {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a is not b:
            acc["other"].append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if path.split(".")[-1] in TEXT_SKIP:
            return
        acc["n"] += 1
        d = 0.0 if (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))) else abs(a - b)
        acc["max"] = max(acc["max"], d)
        if d > 0:
            acc["numdiff"] += 1
    elif isinstance(a, str) and isinstance(b, str):
        if a != b and path.split(".")[-1] not in TEXT_SKIP:
            if prov:
                acc["prov"] += 1
            else:
                acc["str"].append(path)
    elif a != b:
        acc["other"].append((path, type(a).__name__, type(b).__name__))


def main():
    new = pathlib.Path(sys.argv[1])
    ref = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "fase6"
    files = sys.argv[3:] or DEFAULT_FILES
    bad = False
    for f in files:
        if not (new / f).exists():
            print(f"{f}: not produced in {new}")
            continue
        acc = {"n": 0, "max": 0.0, "numdiff": 0, "keys": [], "other": [], "str": [], "prov": 0}
        walk(json.loads((new / f).read_text(encoding="utf-8")), json.loads((ref / f).read_text(encoding="utf-8")), "", acc)
        flag = acc["numdiff"] or acc["keys"] or acc["other"]
        bad |= bool(flag)
        print(f"{f}: numeric leaves {acc['n']}, differing {acc['numdiff']}, max |diff| {acc['max']:.3e}, "
              f"key/length differences {len(acc['keys'])}, bool/null/type differences {len(acc['other'])}, "
              f"other string leaves differing {len(acc['str'])}, provenance fields skipped {acc['prov']}")
        for k in acc["keys"][:5]:
            print("   key:", k)
        for k in acc["other"][:5]:
            print("   other:", k)
        for k in acc["str"][:40]:
            print("   str:", k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
