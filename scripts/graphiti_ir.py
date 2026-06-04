"""
graphiti_ir.py – Emit and parse a custom MLIR-like intermediate representation
for graphiti graphs, capturing all information present in the DOT-based format
while treating DOT files solely as external/output artifacts (never as Python
inputs for the Lean→Python intermediate exchange).

Text format
-----------
// graphiti-ir v1.0
module {
  graphiti.circuit {
    %"arg" = graphiti.node {type = "Entry", bbID = 1, in = "in1:32", out = "out1:32", tagged = false, taggers_num = 0, tagger_id = -1}
    %"mul_0" = graphiti.node {type = "Operator", op = "mul_op", bbID = 1, in = "in1:32 in2:32 ", out = "out1:32 ", delay = 0.000, latency = 4, II = 1, tagged = false, taggers_num = 0, tagger_id = -1}
    graphiti.connect %"arg"#out1 -> %"mul_0"#in1
    graphiti.connect %"arg"#out2 -> %"mul_0"#in2
  }
}

networkx value convention (mirrors graphiti_conv.parse_dot output)
------------------------------------------------------------------
Node IDs are plain Python strings without any surrounding quotes.
Attribute values that appear quoted in the DOT source (type, in, out, op,
value, control, memory, color, mem_address, graphiti_metadata, …) are stored
with their surrounding double-quotes intact, e.g. '"Entry"'.
Attribute values that appear unquoted in the DOT source (bbID, bbcount,
ldcount, stcount, II, latency, delay, tagger_id, taggers_num, tagged, offset,
portId) are stored as plain strings, e.g. '0', 'false', '0.366'.
Edge 'from' and 'to' values mirror the DOT convention: '"out1"', '"in2"'.
"""

import re
import networkx as nx

# ---------------------------------------------------------------------------
# Attribute quoting helpers
# ---------------------------------------------------------------------------

def _emit_attr_val(val) -> str:
    """Emit a networkx attribute value as MLIR text.

    Networkx attribute values are returned verbatim.  DOT-quoted values
    (stored with surrounding double-quotes by pydot, e.g. '"Entry"',
    '"in1:32"', '"{\\"k\\": \\"v\\"}"') already carry the correct MLIR
    representation, including any inner backslash-escape sequences that
    pydot preserved.  Unquoted values (e.g. '0', 'false', '-1') are also
    returned unchanged.
    """
    return str(val)


def _parse_mlir_val(mlir_val: str) -> str:
    """Reconstruct a networkx attribute value from a parsed MLIR attribute value.

    The MLIR scanner returns attribute values verbatim (outer double-quotes
    included, inner backslash-escape sequences preserved), which is exactly
    the representation that pydot stores in networkx for DOT-quoted attributes.
    Bare (unquoted) tokens are returned unchanged.
    """
    return mlir_val


# ---------------------------------------------------------------------------
# Attribute block emitter
# ---------------------------------------------------------------------------

def _emit_attrs(data: dict, skip_keys=()) -> str:
    """Format a dict of node attributes as an MLIR attribute block string."""
    parts = []
    for key, val in data.items():
        if key in skip_keys:
            continue
        parts.append(f'{key} = {_emit_attr_val(val)}')
    return '{' + ', '.join(parts) + '}'


# ---------------------------------------------------------------------------
# Attribute block parser
# ---------------------------------------------------------------------------

def _parse_attrs_str(s: str) -> dict:
    """Parse an MLIR attribute block string into a dict.

    Handles quoted string values (including those containing commas or nested
    braces) by scanning character-by-character with backslash-escape awareness.
    """
    s = s.strip()
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()

    attrs = {}
    while s:
        # Parse key (identifier)
        m = re.match(r'(\w+)\s*=\s*', s)
        if not m:
            break
        key = m.group(1)
        s = s[m.end():]

        if s.startswith('"'):
            # Quoted value: scan for the closing unescaped double-quote
            i = 1
            while i < len(s):
                if s[i] == '\\' and i + 1 < len(s):
                    i += 2  # skip backslash-escaped character
                elif s[i] == '"':
                    break
                else:
                    i += 1
            mlir_val = s[:i + 1]
            s = s[i + 1:].lstrip(', ')
        else:
            # Bare (unquoted) value: everything up to the next comma or end
            m2 = re.match(r'([^,}]+)', s)
            if m2:
                mlir_val = m2.group(1).strip()
                s = s[m2.end():].lstrip(', ')
            else:
                break

        attrs[key] = _parse_mlir_val(mlir_val)

    return attrs


# ---------------------------------------------------------------------------
# Public API: nx_to_mlir / write_mlir
# ---------------------------------------------------------------------------

def nx_to_mlir(nx_graph: nx.MultiDiGraph) -> str:
    """Convert a networkx MultiDiGraph to MLIR-like IR text.

    The resulting text captures all node attributes and edge connectivity
    (including port names) in a format that round-trips through mlir_to_nx.
    """
    lines = ['// graphiti-ir v1.0', 'module {', '  graphiti.circuit {']

    for node_id, data in nx_graph.nodes(data=True):
        attrs_str = _emit_attrs(data)
        lines.append(f'    %"{node_id}" = graphiti.node {attrs_str}')

    for src, dst, data in nx_graph.edges(data=True):
        from_port = str(data.get('from', '""')).strip('"')
        to_port = str(data.get('to', '""')).strip('"')
        extra = {k: v for k, v in data.items() if k not in ('from', 'to', 'key')}
        if extra:
            extra_str = ' ' + _emit_attrs(extra)
        else:
            extra_str = ''
        lines.append(f'    graphiti.connect %"{src}"#{from_port} -> %"{dst}"#{to_port}{extra_str}')

    lines += ['  }', '}']
    return '\n'.join(lines) + '\n'


def write_mlir(output_path: str, nx_graph: nx.MultiDiGraph) -> None:
    """Write a networkx graph to a file as MLIR-like IR text."""
    with open(output_path, 'w') as f:
        f.write(nx_to_mlir(nx_graph))


# ---------------------------------------------------------------------------
# Public API: mlir_to_nx / parse_mlir
# ---------------------------------------------------------------------------

# Regex patterns for the two statement types
_NODE_RE = re.compile(
    r'^\s*%"([^"]+)"\s*=\s*graphiti\.node\s*(\{.*\})\s*$'
)
_CONNECT_RE = re.compile(
    r'^\s*graphiti\.connect\s+%"([^"]+)"#(\S+)\s+->\s+%"([^"]+)"#(\S+)(?:\s+(\{.*\}))?\s*$'
)


def mlir_to_nx(text: str) -> nx.MultiDiGraph:
    """Parse graphiti MLIR-like IR text into a networkx MultiDiGraph.

    The returned graph has the same node-ID and attribute-quoting conventions
    as the graph returned by graphiti_conv.parse_dot, so downstream code
    (graphiti-to-dynamatic.py) can operate unchanged.
    """
    g = nx.MultiDiGraph()

    for line in text.splitlines():
        m = _NODE_RE.match(line)
        if m:
            node_id = m.group(1)
            attrs = _parse_attrs_str(m.group(2))
            g.add_node(node_id, **attrs)
            continue

        m = _CONNECT_RE.match(line)
        if m:
            src, src_port, dst, dst_port, attrs_str = m.groups()
            edge_attrs = {'from': f'"{src_port}"', 'to': f'"{dst_port}"'}
            if attrs_str:
                edge_attrs.update(_parse_attrs_str(attrs_str))
            g.add_edge(src, dst, **edge_attrs)

    return g


def parse_mlir(input_path: str) -> nx.MultiDiGraph:
    """Read a graphiti MLIR-like IR file and return a networkx MultiDiGraph.

    Drop-in replacement for graphiti_conv.parse_dot for consumers of the
    Lean→Python intermediate representation.
    """
    with open(input_path) as f:
        return mlir_to_nx(f.read())
