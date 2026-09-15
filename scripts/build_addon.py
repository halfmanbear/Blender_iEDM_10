"""Build dist/iEDM_10-<version>.zip from git-tracked add-on files only."""
import ast
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "iEDM_10"


def read_version():
    tree = ast.parse((ROOT / PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "bl_info" for t in node.targets):
            return ".".join(str(v) for v in ast.literal_eval(node.value)["version"])
    sys.exit("bl_info not found in {}/__init__.py".format(PACKAGE))


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z", PACKAGE], cwd=ROOT, check=True, capture_output=True).stdout
    return sorted(p for p in out.decode("utf-8").split("\0") if p.endswith(".py"))


def main():
    files = tracked_files()
    if not files:
        sys.exit("no tracked files under {}/".format(PACKAGE))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / "{}-{}.zip".format(PACKAGE, read_version())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(ROOT / rel, rel)
        zf.write(ROOT / "LICENCE", "{}/LICENCE".format(PACKAGE))
    print("Built {} ({} files)".format(target.relative_to(ROOT), len(files) + 1))


if __name__ == "__main__":
    main()
