"""
test_graphiti_ir.py – Round-trip tests for graphiti_ir.py.

Tests verify that:
  1. DOT files can be parsed with graphiti_conv.parse_dot and then round-tripped
     through graphiti_ir.nx_to_mlir / graphiti_ir.mlir_to_nx with all node
     attributes and edge connectivity preserved.
  2. The MLIR text format is well-formed (correct structure, quoting, etc.).
  3. Attribute-quoting conventions match those expected by graphiti-to-dynamatic.py.
"""

import sys
import os
import networkx as nx

# Resolve the scripts directory so this file can be run from any location.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(os.path.dirname(_SCRIPTS_DIR), 'tests')
sys.path.insert(0, _SCRIPTS_DIR)

import graphiti_conv as gc
import graphiti_ir as gi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_attrs(g: nx.MultiDiGraph, nid: str) -> dict:
    return dict(g.nodes[nid])

def _edge_set(g: nx.MultiDiGraph):
    """Return a frozenset of (src, dst, from_attr, to_attr) tuples."""
    return frozenset(
        (u, v, d.get('from', ''), d.get('to', ''))
        for u, v, d in g.edges(data=True)
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_round_trip_no_control_flow():
    """Round-trip the no-control-flow DOT through MLIR and back."""
    dot_path = os.path.join(_TESTS_DIR, 'dynamatic-no-control-flow.dot')
    original = gc.parse_dot(dot_path)

    mlir_text = gi.nx_to_mlir(original)

    # Verify MLIR structure
    assert '// graphiti-ir v1.0' in mlir_text
    assert 'module {' in mlir_text
    assert 'graphiti.circuit {' in mlir_text
    assert 'graphiti.node' in mlir_text
    assert 'graphiti.connect' in mlir_text

    restored = gi.mlir_to_nx(mlir_text)

    # Node sets must match
    assert set(original.nodes()) == set(restored.nodes()), (
        f"Node mismatch: {set(original.nodes()) ^ set(restored.nodes())}"
    )

    # Every node's attributes must be preserved exactly
    for nid in original.nodes():
        orig_attrs = _node_attrs(original, nid)
        rest_attrs = _node_attrs(restored, nid)
        assert orig_attrs == rest_attrs, (
            f"Attribute mismatch for node '{nid}':\n"
            f"  original : {orig_attrs}\n"
            f"  restored : {rest_attrs}"
        )

    # Edge connectivity (from/to port names) must be preserved
    assert _edge_set(original) == _edge_set(restored), (
        f"Edge mismatch"
    )


def test_round_trip_if_then_else_mux():
    """Round-trip the if-then-else-mux DOT through MLIR and back."""
    dot_path = os.path.join(_TESTS_DIR, 'dynamatic-if-then-else-mux.dot')
    original = gc.parse_dot(dot_path)

    mlir_text = gi.nx_to_mlir(original)
    restored = gi.mlir_to_nx(mlir_text)

    assert set(original.nodes()) == set(restored.nodes())
    for nid in original.nodes():
        assert _node_attrs(original, nid) == _node_attrs(restored, nid), (
            f"Attribute mismatch for node '{nid}'"
        )
    assert _edge_set(original) == _edge_set(restored)


def test_quoting_conventions():
    """Verify that the quoting conventions match graphiti_conv expectations."""
    g = nx.MultiDiGraph()
    # Node with mixed quoted/unquoted attrs (mirrors what graphiti_conv.parse_dot returns)
    g.add_node('arg', **{
        'type': '"Entry"',
        'bbID': '1',
        'in': '"in1:32"',
        'out': '"out1:32"',
        'tagged': 'false',
        'taggers_num': '0',
        'tagger_id': '-1',
    })
    g.add_node('fork_0', **{
        'type': '"Fork"',
        'bbID': '1',
        'in': '"in1:32"',
        'out': '"out1:32 out2:32"',
        'tagged': 'false',
        'taggers_num': '0',
        'tagger_id': '-1',
    })
    g.add_edge('arg', 'fork_0', **{'from': '"out1"', 'to': '"in1"'})

    mlir = gi.nx_to_mlir(g)

    # Quoted attrs must appear with quotes in the MLIR text
    assert 'type = "Entry"' in mlir
    assert 'in = "in1:32"' in mlir
    assert 'out = "out1:32"' in mlir
    # Unquoted attrs must appear without quotes
    assert 'bbID = 1' in mlir
    assert 'tagged = false' in mlir
    assert 'tagger_id = -1' in mlir
    # Port names in connect line must appear without surrounding quotes
    assert '#out1 ->' in mlir
    assert '-> %"fork_0"#in1' in mlir

    restored = gi.mlir_to_nx(mlir)

    # Verify quoting is reconstructed correctly
    assert restored.nodes['arg']['type'] == '"Entry"'
    assert restored.nodes['arg']['bbID'] == '1'
    assert restored.nodes['arg']['in'] == '"in1:32"'
    assert restored.nodes['arg']['tagged'] == 'false'
    # Edge attrs
    edges = list(restored.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]['from'] == '"out1"'
    assert edges[0][2]['to'] == '"in1"'


def test_graphiti_metadata_round_trip():
    """Verify that graphiti_metadata (nested JSON) survives a round-trip.

    After parse_dot, pydot stores quoted attribute values verbatim, including
    the surrounding double-quotes and any inner backslash-escape sequences.
    For graphiti_metadata the DOT file contains:
      graphiti_metadata = "{\"parent_mc\": \"MC_A\", \"shard_num\": 1}"
    so pydot returns the networkx value with the outer '"' kept and the inner
    \\" sequences preserved – exactly as if we had written:
      '"' + inner_json.replace('"', '\\"') + '"'
    The MLIR round-trip must reproduce this same value so that the double
    json.loads pattern in graphiti-to-dynamatic.py continues to work.
    """
    import json
    inner_json = json.dumps({'parent_mc': 'MC_A', 'shard_num': 1})
    # Construct the networkx value as pydot would return it:
    # outer '"' plus the JSON string with inner '"' escaped as '\"'
    pydot_value = '"' + inner_json.replace('"', '\\"') + '"'

    g = nx.MultiDiGraph()
    g.add_node('mc_shard', **{
        'type': '"MC"',
        'bbID': '0',
        'in': '"in1:32*l0a"',
        'out': '"out1:32*l0d out2:0*e"',
        'graphiti_metadata': pydot_value,
    })

    mlir = gi.nx_to_mlir(g)
    restored = gi.mlir_to_nx(mlir)

    orig = g.nodes['mc_shard']
    rest = restored.nodes['mc_shard']
    assert orig == rest, f"Mismatch:\n  orig={orig}\n  rest={rest}"

    # The double json.loads pattern used in graphiti-to-dynamatic.py must work
    raw = rest['graphiti_metadata']
    step1 = json.loads(raw)          # strips outer networkx/DOT quotes
    step2 = json.loads(step1)        # parses the inner JSON
    assert step2 == {'parent_mc': 'MC_A', 'shard_num': 1}


def test_write_and_parse_mlir(tmp_path):
    """write_mlir / parse_mlir round-trip via a real file."""
    g = nx.MultiDiGraph()
    g.add_node('n', **{'type': '"Fork"', 'bbID': '2', 'in': '"in1:32"', 'out': '"out1:32 out2:32"', 'tagged': 'false', 'taggers_num': '0', 'tagger_id': '-1'})
    g.add_node('m', **{'type': '"Sink"', 'bbID': '2', 'in': '"in1:32"'})
    g.add_edge('n', 'm', **{'from': '"out1"', 'to': '"in1"'})

    path = str(tmp_path / 'test.gir')
    gi.write_mlir(path, g)

    restored = gi.parse_mlir(path)
    assert set(g.nodes()) == set(restored.nodes())
    for nid in g.nodes():
        assert dict(g.nodes[nid]) == dict(restored.nodes[nid])
    assert _edge_set(g) == _edge_set(restored)


# ---------------------------------------------------------------------------
# Runner (also compatible with pytest)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tempfile, pathlib

    print('test_round_trip_no_control_flow ...', end=' ', flush=True)
    test_round_trip_no_control_flow()
    print('OK')

    print('test_round_trip_if_then_else_mux ...', end=' ', flush=True)
    test_round_trip_if_then_else_mux()
    print('OK')

    print('test_quoting_conventions ...', end=' ', flush=True)
    test_quoting_conventions()
    print('OK')

    print('test_graphiti_metadata_round_trip ...', end=' ', flush=True)
    test_graphiti_metadata_round_trip()
    print('OK')

    with tempfile.TemporaryDirectory() as td:
        print('test_write_and_parse_mlir ...', end=' ', flush=True)
        test_write_and_parse_mlir(pathlib.Path(td))
        print('OK')

    print('\nAll tests passed.')
