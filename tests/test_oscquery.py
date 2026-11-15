import json

from fbt.oscquery import VRSYSTEM_ENDPOINTS, build_tree


def _csharp_lookup(tree):
    """Mimic vrc-oscquery-lib RebuildLookup: walk CONTENTS, key by FULL_PATH."""
    out = {"/": tree}

    def walk(n):
        for child in (n.get("CONTENTS") or {}).values():
            assert child["FULL_PATH"] not in out  # C# Dictionary.Add would throw
            out[child["FULL_PATH"]] = child
            walk(child)
    walk(tree)
    return out


def test_tree_has_exact_vrsystem_paths_like_vrchat_sample():
    tree, _ = build_tree()
    lookup = _csharp_lookup(json.loads(json.dumps(tree)))
    for ep in VRSYSTEM_ENDPOINTS:
        assert lookup[ep]["TYPE"] == "ffffff" and lookup[ep]["ACCESS"] == 2
    for p in ("/tracking", "/tracking/vrsystem", "/tracking/vrsystem/head"):
        assert p in lookup
    assert "null" not in json.dumps(tree)
