import json


def write_compact_mapping(path, mapping):
    """Write a JSON object with one compact value per key."""
    lines = ["{"]
    items = list(mapping.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        compact_value = json.dumps(value, separators=(',', ': '), ensure_ascii=False)
        lines.append(f"  {json.dumps(key)}: {compact_value}{comma}")
    lines.append("}")
    content = "\n".join(lines) + "\n"
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != content:
        path.write_text(content, encoding="utf-8")


def write_compact_result_array(path, results):
    """Write only a JSON array of compact measurement records."""
    lines = ["["]
    for i, record in enumerate(results):
        comma = "," if i < len(results) - 1 else ""
        lines.append(f"  {json.dumps(record, separators=(',', ': '), ensure_ascii=False)}{comma}")
    lines.append("]")
    content = "\n".join(lines) + "\n"
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != content:
        path.write_text(content, encoding="utf-8")
