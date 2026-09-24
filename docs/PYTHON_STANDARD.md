# Python coding standard

This standard adapts the reliability, testability, and readability goals of the
[JSF Air Vehicle C++ Coding Standards](https://stroustrup.com/JSF-AV-rules.pdf)
to this Blender importer. It is a project maintenance policy, not a safety
certification. C++ rules about pointers, memory layout, templates, and
preprocessing do not apply to Python and are not copied here.

## Required rules

1. **Bound size and complexity.** Keep each Python file at or below 400 physical
   lines and each function or method at or below 100 logical lines. The checker
   counts physical lines with `splitlines()` and logical lines using Python's
   tokenizer (`NEWLINE` tokens), including a function's signature and nested
   statements. Keep Ruff's cyclomatic complexity limit of 15. Split by
   responsibility, not by arbitrary line ranges.
2. **Preserve contracts.** Refactors must retain import paths, Blender operator
   behavior, EDM parser registration, binary interpretation, and exporter
   metadata unless a behavior change is explicitly specified and tested.
3. **Make failures visible.** Validate untrusted binary counts and indices
   before allocating or indexing. Raise a useful exception when parsing cannot
   continue. Catch only expected exceptions at a boundary; when Blender
   compatibility requires a fallback, document and log the fallback.
4. **Keep effects local.** Avoid new module-level mutable state. Pass values
   explicitly where practical, and keep Blender scene writes in importer
   modules rather than pure format or math helpers.
5. **Test behavior.** Add or update a focused regression when changing binary
   parsing, transforms, animation, materials, lights, or public import paths.
   Check both supported Blender versions for version-sensitive behavior.
6. **Pass automated checks.** Run `python scripts/check_code_standard.py`
   before merging. Its Ruff configuration is in `pyproject.toml`; avoid broad
   `noqa` and `type: ignore` comments. Any suppression must name the specific
   rule and explain the interoperability reason beside the code.

## Advisory rules

- Name functions for the action or value they perform. Keep unrelated object
  creation, metadata stamping, and animation setup in separate modules.
- Type pure Python boundaries and data structures. Use `Any` at Blender's
  dynamic API boundary only when a precise type is unavailable, and keep that
  boundary narrow.
- Prefer straightforward control flow and early returns. Avoid hidden imports,
  broad exception handling, and helpers that depend on injected module globals.

## Existing debt and exceptions

`scripts/code_standard_baseline.json` records pre-existing files and functions
above the limits and Ruff findings by file and rule. New files and functions
must meet the limits. Existing exceptions may remain unchanged or shrink, but
must not grow. When a file or function is brought within the limit, remove its
baseline entry; do not increase an entry to accommodate new code. The baseline
is an inventory for migration, not a general exemption.

Responsibility-based module splits move existing Ruff findings to their new
module paths. Such migrations must remove the corresponding allowance from
the original file and must not increase the combined count for any rule.
New code receives no additional allowance.

For a necessary exception that cannot be eliminated in the current change,
state the concrete reason and an exit condition in the review description, and
record the specific exception in this document. Do not suppress a rule without
an explanation. As of this standard's adoption, there are no new exceptions.

All project Python files now meet the 400-physical-line limit, and
`legacy_files` is empty. No file-size exceptions remain. The remaining Ruff
entries are the measured follow-up backlog; they do not exempt files from
the size limit.
