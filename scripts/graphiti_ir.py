#!/usr/bin/env python3
"""
graphiti_ir.py – Emitter for a custom MLIR-like IR for Graphiti dataflow graphs.

This module defines:
  - A type system that captures all port qualifiers found in Graphiti/Dynamatic
    dot files (?, +, -, *i, *e, *l{n}a, *l{n}d, *s{n}a, *s{n}d, *c{n}).
  - Data structures (Value, Port, Operand, EdgeAttrs, Operation, BasicBlock,
    GraphitiModule) that faithfully represent every piece of information carried
    in those dot files.
  - An MLIR-inspired text emitter.
  - A NodeBuilder factory with one method per Dynamatic/Graphiti component type.
  - Example programs that construct all test graphs programmatically (no dot
    files are read at runtime).

Text format overview
--------------------
    graphiti.module @circuit {

      ^bb1  // block1
        %arg.out1 = graphiti.entry() {bbID = 1, tagged = false, …} : () -> i32
        %fork_0.out1, %fork_0.out2 = graphiti.fork(%arg.out1 : in1)
                                     {bbID = 1, tagged = false, …}
                                     : (i32) -> (i32, i32)

      ^bb0
        %end_0.out1 = graphiti.exit(%ret_0.out1 : in1) {bbID = 0} : (i32) -> i32
    }

Each SSA value name encodes the source node and output-port name:
    %<node>.<port>

Operand annotations capture the destination port name and optional edge
metadata (color, mem_address, minlen):
    %src.out1 : in2 [color = "red", mem_address = false]
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union

# ---------------------------------------------------------------------------
# Port qualifiers and type system
# ---------------------------------------------------------------------------

class PortQualifier(Enum):
    """Semantic qualifier for a plain port (no numeric suffix)."""
    NONE         = ""     # ordinary data port
    OPTIONAL     = "?"    # conditional / selector port
    TRUE_BRANCH  = "+"    # true-branch output of a Branch node
    FALSE_BRANCH = "-"    # false-branch output of a Branch node
    INIT_TOKEN   = "*i"   # initial token on a loop-back arc
    END_TOKEN    = "*e"   # end-of-memory token from an MC node


@dataclass(frozen=True)
class LoadAddrQual:
    """Port carries a load address: *l{port_num}a."""
    port_num: int

    def __str__(self) -> str:
        return f"*l{self.port_num}a"


@dataclass(frozen=True)
class LoadDataQual:
    """Port carries load data: *l{port_num}d."""
    port_num: int

    def __str__(self) -> str:
        return f"*l{self.port_num}d"


@dataclass(frozen=True)
class StoreAddrQual:
    """Port carries a store address: *s{port_num}a."""
    port_num: int

    def __str__(self) -> str:
        return f"*s{self.port_num}a"


@dataclass(frozen=True)
class StoreDataQual:
    """Port carries store data: *s{port_num}d."""
    port_num: int

    def __str__(self) -> str:
        return f"*s{self.port_num}d"


@dataclass(frozen=True)
class CountQual:
    """Port carries a count/control token: *c{port_num}."""
    port_num: int

    def __str__(self) -> str:
        return f"*c{self.port_num}"


# Union of all qualifier types.
PortQual = Union[
    PortQualifier,
    LoadAddrQual, LoadDataQual,
    StoreAddrQual, StoreDataQual,
    CountQual,
]


@dataclass(frozen=True)
class IntType:
    """
    An integer type with a bit-width and an optional port qualifier.

    Examples:
        IntType(32)                          ->  i32
        IntType(1, PortQualifier.OPTIONAL)   ->  i1?
        IntType(32, PortQualifier.INIT_TOKEN)->  i32*i
        IntType(32, LoadAddrQual(0))         ->  i32*l0a
        IntType(0, PortQualifier.END_TOKEN)  ->  i0*e
    """
    width: int
    qualifier: PortQual = PortQualifier.NONE

    def __str__(self) -> str:
        q = self.qualifier
        if isinstance(q, PortQualifier):
            return f"i{self.width}{q.value}"
        return f"i{self.width}{q}"

    def base(self) -> "IntType":
        """Return the same type without any qualifier."""
        return IntType(self.width)


# Convenience constructors --------------------------------------------------

def i(n: int) -> IntType:
    """Plain integer type."""
    return IntType(n)


def i_opt(n: int) -> IntType:
    """Optional/conditional input (Mux selector, Branch condition)."""
    return IntType(n, PortQualifier.OPTIONAL)


def i_true(n: int) -> IntType:
    """True-branch output."""
    return IntType(n, PortQualifier.TRUE_BRANCH)


def i_false(n: int) -> IntType:
    """False-branch output."""
    return IntType(n, PortQualifier.FALSE_BRANCH)


def i_init(n: int) -> IntType:
    """Initial token on a loop-back arc."""
    return IntType(n, PortQualifier.INIT_TOKEN)


def i_end(n: int) -> IntType:
    """End-of-memory token."""
    return IntType(n, PortQualifier.END_TOKEN)


def i_laddr(n: int, port: int) -> IntType:
    """Load address token for load port *port*."""
    return IntType(n, LoadAddrQual(port))


def i_ldata(n: int, port: int) -> IntType:
    """Load data token for load port *port*."""
    return IntType(n, LoadDataQual(port))


def i_saddr(n: int, port: int) -> IntType:
    """Store address token for store port *port*."""
    return IntType(n, StoreAddrQual(port))


def i_sdata(n: int, port: int) -> IntType:
    """Store data token for store port *port*."""
    return IntType(n, StoreDataQual(port))


def i_count(n: int, port: int) -> IntType:
    """Count/control token for store port *port*."""
    return IntType(n, CountQual(port))


# ---------------------------------------------------------------------------
# Parsing port specs from dot-file attribute strings (utility)
# ---------------------------------------------------------------------------

_PORT_SPEC_RE = re.compile(
    r"(\w+)"          # port name (e.g. in1, out2, in1?)
    r"([?+\-]?)"      # optional port-level qualifier
    r":(\d+)"         # bit-width
    r"(\*[a-z0-9]+)?" # optional *-suffix (e.g. *i, *e, *l0a, *s1d, *c0)
)


def parse_port_spec(spec: str) -> "Port":
    """
    Parse a single port specification string (as used in dot ``in``/``out`` attrs).

    Examples::
        parse_port_spec("in1:32")        -> Port("in1", IntType(32))
        parse_port_spec("in2?:1")        -> Port("in2", IntType(1, OPTIONAL))
        parse_port_spec("out1+:32")      -> Port("out1", IntType(32, TRUE_BRANCH))
        parse_port_spec("out2-:32")      -> Port("out2", IntType(32, FALSE_BRANCH))
        parse_port_spec("in1:0*i")       -> Port("in1", IntType(0, INIT_TOKEN))
        parse_port_spec("in1:0*e")       -> Port("in1", IntType(0, END_TOKEN))
        parse_port_spec("in1:32*l0a")    -> Port("in1", IntType(32, LoadAddrQual(0)))
        parse_port_spec("out1:32*l0d")   -> Port("out1", IntType(32, LoadDataQual(0)))
        parse_port_spec("in2:32*s0a")    -> Port("in2", IntType(32, StoreAddrQual(0)))
        parse_port_spec("in3:32*s0d")    -> Port("in3", IntType(32, StoreDataQual(0)))
        parse_port_spec("in1:32*c0")     -> Port("in1", IntType(32, CountQual(0)))
    """
    spec = spec.strip()
    m = _PORT_SPEC_RE.fullmatch(spec)
    if not m:
        raise ValueError(f"Cannot parse port spec: {spec!r}")
    name, pq_char, width_str, suffix = m.groups()
    width = int(width_str)

    # Determine port-level qualifier (?, +, -)
    if pq_char == "?":
        base_qual: PortQual = PortQualifier.OPTIONAL
    elif pq_char == "+":
        base_qual = PortQualifier.TRUE_BRANCH
    elif pq_char == "-":
        base_qual = PortQualifier.FALSE_BRANCH
    else:
        base_qual = PortQualifier.NONE

    # Determine *-suffix qualifier (overrides base if present)
    if suffix:
        s = suffix[1:]  # strip leading '*'
        if s == "i":
            qual: PortQual = PortQualifier.INIT_TOKEN
        elif s == "e":
            qual = PortQualifier.END_TOKEN
        elif re.fullmatch(r"l(\d+)a", s):
            qual = LoadAddrQual(int(re.fullmatch(r"l(\d+)a", s).group(1)))
        elif re.fullmatch(r"l(\d+)d", s):
            qual = LoadDataQual(int(re.fullmatch(r"l(\d+)d", s).group(1)))
        elif re.fullmatch(r"s(\d+)a", s):
            qual = StoreAddrQual(int(re.fullmatch(r"s(\d+)a", s).group(1)))
        elif re.fullmatch(r"s(\d+)d", s):
            qual = StoreDataQual(int(re.fullmatch(r"s(\d+)d", s).group(1)))
        elif re.fullmatch(r"c(\d+)", s):
            qual = CountQual(int(re.fullmatch(r"c(\d+)", s).group(1)))
        else:
            raise ValueError(f"Unknown port suffix: *{s!r} in {spec!r}")
    else:
        qual = base_qual

    return Port(name, IntType(width, qual))


def parse_ports(attr_string: str) -> list["Port"]:
    """Parse a space-separated list of port specs (dot ``in``/``out`` attr value)."""
    return [parse_port_spec(s) for s in attr_string.split() if s.strip()]


# ---------------------------------------------------------------------------
# Core IR data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    """A named port on a node, associated with a type."""
    name: str
    type: IntType

    def __str__(self) -> str:
        return f"{self.name}:{self.type}"


@dataclass(frozen=True)
class Value:
    """
    An SSA value representing the token produced by a specific output port of
    a node.  The SSA name is ``%<node_name>.<port_name>``.
    """
    node_name: str
    port_name: str
    type: IntType

    @property
    def ssa_name(self) -> str:
        return f"%{self.node_name}.{self.port_name}"

    def __str__(self) -> str:
        return self.ssa_name


@dataclass
class EdgeAttrs:
    """
    Visual and semantic metadata attached to a graph edge (dot edge attributes).

    Attributes
    ----------
    color       : Edge color hint from the dot file (e.g. ``"red"``, ``"gold3"``,
                  ``"blue"``, ``"darkgreen"``).
    mem_address : True when the edge carries a memory address to/from an MC node.
    minlen      : Layout hint (minimum edge length in the dot renderer).
    """
    color: Optional[str] = None
    mem_address: Optional[bool] = None
    minlen: Optional[int] = None

    def is_empty(self) -> bool:
        return self.color is None and self.mem_address is None and self.minlen is None

    def __str__(self) -> str:
        parts: list[str] = []
        if self.color is not None:
            parts.append(f'color = "{self.color}"')
        if self.mem_address is not None:
            parts.append(f"mem_address = {str(self.mem_address).lower()}")
        if self.minlen is not None:
            parts.append(f"minlen = {self.minlen}")
        return ", ".join(parts)


@dataclass
class Operand:
    """
    A use of an SSA value as an operation input, annotated with:
      - the name of the *destination* port on the receiving node (the ``to``
        attribute on a dot edge),
      - optional edge-level metadata preserved from the dot file.
    """
    value: Value
    dest_port: str
    edge_attrs: EdgeAttrs = field(default_factory=EdgeAttrs)

    def __str__(self) -> str:
        s = f"{self.value.ssa_name} : {self.dest_port}"
        if not self.edge_attrs.is_empty():
            s += f" [{self.edge_attrs}]"
        return s


# ---------------------------------------------------------------------------
# Attribute formatting
# ---------------------------------------------------------------------------

def _fmt_attr_val(v: object) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        # Use up to 3 decimal places, strip trailing zeros.
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _fmt_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    pairs = [f"{k} = {_fmt_attr_val(v)}" for k, v in attrs.items()]
    return "{" + ", ".join(pairs) + "}"


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

_DIALECT = "graphiti"


@dataclass
class Operation:
    """
    A single dataflow node, emitted as an MLIR-like operation.

    Emitted syntax::

        %n.out1, %n.out2 = graphiti.<kind>(%a.out1 : in1, %b.out1 : in2)
                           {attr1 = val1, attr2 = val2}
                           : (i32, i32) -> (i32, i32)

    Parameters
    ----------
    node_name : Unique name for this node (becomes the prefix of SSA names).
    kind      : Operation name within the ``graphiti`` dialect (e.g. ``"fork"``).
    operands  : Ordered list of :class:`Operand` instances.
    out_ports : Ordered list of output :class:`Port` instances.
    attrs     : Node-level attribute dictionary.
    """
    node_name: str
    kind: str
    operands: list[Operand]
    out_ports: list[Port]
    attrs: dict

    def result_values(self) -> list[Value]:
        """Return SSA Values for all output ports."""
        return [Value(self.node_name, p.name, p.type) for p in self.out_ports]

    def result(self, port_name: str) -> Value:
        """Return the SSA Value for a specific output port by name."""
        for p in self.out_ports:
            if p.name == port_name:
                return Value(self.node_name, p.name, p.type)
        raise KeyError(f"No output port '{port_name}' on '{self.node_name}'")

    def emit(self, indent: str = "    ") -> str:
        """Emit the MLIR-like text for this operation."""
        # Left-hand side
        results = self.result_values()
        lhs = (", ".join(v.ssa_name for v in results) + " = ") if results else ""

        op_name = f"{_DIALECT}.{self.kind}"

        operand_str = "(" + ", ".join(str(o) for o in self.operands) + ")"

        attr_str = (" " + _fmt_attrs(self.attrs)) if self.attrs else ""

        # Type signature
        in_types_str = ", ".join(str(o.value.type) for o in self.operands)
        out_ports = self.out_ports
        if len(out_ports) == 0:
            out_types_str = "()"
        elif len(out_ports) == 1:
            out_types_str = str(out_ports[0].type)
        else:
            out_types_str = "(" + ", ".join(str(p.type) for p in out_ports) + ")"

        type_sig = f" : ({in_types_str}) -> {out_types_str}"

        line = f"{lhs}{op_name}{operand_str}{attr_str}{type_sig}"
        full = indent + line

        # If the line exceeds 100 characters, wrap at the type-signature boundary.
        if len(full) > 100:
            head = indent + f"{lhs}{op_name}{operand_str}{attr_str}"
            cont = indent + "        " + type_sig.lstrip()
            return head + "\n" + cont
        return full


# ---------------------------------------------------------------------------
# Basic block
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    """
    A basic block groups operations that belong to the same control-flow block
    (identified by ``bbID`` in the dot files).
    """
    bb_id: int
    label: Optional[str] = None
    operations: list[Operation] = field(default_factory=list)

    def add(self, op: Operation) -> Operation:
        self.operations.append(op)
        return op

    def emit(self) -> str:
        comment = f"  // {self.label}" if self.label else ""
        header = f"  ^bb{self.bb_id}{comment}"
        body = "\n".join(op.emit() for op in self.operations)
        return header + "\n" + body


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@dataclass
class GraphitiModule:
    """Top-level IR container for a Graphiti dataflow graph."""
    name: str
    blocks: list[BasicBlock] = field(default_factory=list)

    def add_block(self, bb: BasicBlock) -> BasicBlock:
        self.blocks.append(bb)
        return bb

    def emit(self) -> str:
        header = f"graphiti.module @{self.name} {{"
        body = "\n\n".join(bb.emit() for bb in self.blocks)
        footer = "}"
        return "\n".join([header, body, footer])

    def print(self) -> None:
        print(self.emit())


# ---------------------------------------------------------------------------
# NodeBuilder – factory methods for all Graphiti/Dynamatic component types
# ---------------------------------------------------------------------------

def _std_tagged_attrs(
    bb_id: int,
    tagged: bool = False,
    taggers_num: int = 0,
    tagger_id: int = -1,
    **extra,
) -> dict:
    d = dict(bbID=bb_id, tagged=tagged, taggers_num=taggers_num, tagger_id=tagger_id)
    d.update(extra)
    return d


class NodeBuilder:
    """
    Factory methods that construct :class:`Operation` instances for every node
    type found in Graphiti and Dynamatic dot files.

    All methods that accept ``**kw`` pass the extra keyword arguments into the
    operation's attribute dictionary, allowing callers to supply any additional
    dot-file attributes (e.g. ``graphiti_metadata``, ``constants``, …).
    """

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    @staticmethod
    def entry(
        name: str,
        bb_id: int,
        out_type: IntType,
        control: bool = False,
        **kw,
    ) -> Operation:
        """
        Entry node (function argument or control-start token).

        Dot type: ``"Entry"``
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        if control:
            attrs["control"] = True
        return Operation(name, "entry", [], [Port("out1", out_type)], attrs)

    @staticmethod
    def exit(
        name: str,
        bb_id: int,
        operands: list[Operand],
        out_type: IntType,
    ) -> Operation:
        """
        Exit node (function return / program end).

        Dot type: ``"Exit"``
        """
        return Operation(name, "exit", operands, [Port("out1", out_type)], {"bbID": bb_id})

    # ------------------------------------------------------------------
    # Constant
    # ------------------------------------------------------------------

    @staticmethod
    def constant(
        name: str,
        bb_id: int,
        ctrl: Operand,
        value: str,
        out_type: IntType,
        **kw,
    ) -> Operation:
        """
        Constant node.

        Dot type: ``"Constant"``

        Parameters
        ----------
        ctrl      : Control token operand (triggers output).
        value     : Hex literal string (e.g. ``"0x00000002"``).
        out_type  : Output token type.
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        attrs["value"] = value
        return Operation(name, "constant", [ctrl], [Port("out1", out_type)], attrs)

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    @staticmethod
    def fork(
        name: str,
        bb_id: int,
        operand: Operand,
        n_out: int,
        **kw,
    ) -> Operation:
        """
        Fork node (1-to-N replicator).

        Dot type: ``"Fork"`` / ``"fork Bool 2"``
        """
        t = IntType(operand.value.type.width)  # output has no qualifier
        out_ports = [Port(f"out{i + 1}", t) for i in range(n_out)]
        attrs = _std_tagged_attrs(bb_id, **kw)
        return Operation(name, "fork", [operand], out_ports, attrs)

    @staticmethod
    def fork_ports(
        name: str,
        bb_id: int,
        operand: Operand,
        out_ports: list[Port],
        **kw,
    ) -> Operation:
        """
        Fork node with explicit (potentially non-contiguous) output port specs.
        Use when the dot file has non-standard port names (e.g. ``out0``-indexed).
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        return Operation(name, "fork", [operand], out_ports, attrs)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def merge(
        name: str,
        bb_id: int,
        operands: list[Operand],
        out_type: IntType,
        delay: Optional[float] = None,
        **kw,
    ) -> Operation:
        """
        Merge node (N-to-1 non-deterministic selector).

        Dot type: ``"Merge"``
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        if delay is not None:
            attrs["delay"] = delay
        return Operation(name, "merge", operands, [Port("out1", out_type)], attrs)

    # ------------------------------------------------------------------
    # Mux
    # ------------------------------------------------------------------

    @staticmethod
    def mux(
        name: str,
        bb_id: int,
        sel: Operand,
        data_inputs: list[Operand],
        out_type: IntType,
        delay: float = 0.366,
        **kw,
    ) -> Operation:
        """
        Mux node (deterministic selector driven by *sel*).

        Dot type: ``"Mux"`` / ``"mux T"``

        The selector operand comes first (as ``in1?``); data inputs follow.
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        attrs["delay"] = delay
        return Operation(name, "mux", [sel] + data_inputs, [Port("out1", out_type)], attrs)

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------

    @staticmethod
    def branch(
        name: str,
        bb_id: int,
        data: Operand,
        cond: Operand,
        data_width: int,
        **kw,
    ) -> Operation:
        """
        Branch node (conditional routing).

        Dot type: ``"Branch"`` / ``"branch T"``

        ``out1+`` carries the token on the true path; ``out2-`` on the false path.
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        return Operation(
            name, "branch", [data, cond],
            [Port("out1", i_true(data_width)), Port("out2", i_false(data_width))],
            attrs,
        )

    # ------------------------------------------------------------------
    # Operator (generic arithmetic / comparison / memory-access node)
    # ------------------------------------------------------------------

    @staticmethod
    def operator(
        name: str,
        bb_id: int,
        op: str,
        operands: list[Operand],
        out_ports: list[Port],
        delay: float = 0.0,
        latency: int = 0,
        II: int = 1,
        **kw,
    ) -> Operation:
        """
        Generic operator node.

        Dot type: ``"Operator"``

        Parameters
        ----------
        op       : Operation mnemonic (e.g. ``"add_op"``, ``"mul_op"``,
                   ``"mc_load_op"``, ``"mc_store_op"``, ``"icmp_sgt_op"``…).
        latency  : Pipeline latency in cycles.
        II       : Initiation interval.
        """
        attrs: dict = {
            "bbID": bb_id, "op": op,
            "delay": delay, "latency": latency, "II": II,
            "tagged": False, "taggers_num": 0, "tagger_id": -1,
        }
        attrs.update(kw)
        return Operation(name, "operator", operands, out_ports, attrs)

    # ------------------------------------------------------------------
    # Sink
    # ------------------------------------------------------------------

    @staticmethod
    def sink(name: str, bb_id: int, operand: Operand) -> Operation:
        """
        Sink node (discards its input token).

        Dot type: ``"Sink"``
        """
        return Operation(name, "sink", [operand], [], {"bbID": bb_id})

    # ------------------------------------------------------------------
    # Memory Controller (MC)
    # ------------------------------------------------------------------

    @staticmethod
    def mc(
        name: str,
        operands: list[Operand],
        out_ports: list[Port],
        memory: str,
        bbcount: int = 0,
        ldcount: int = 0,
        stcount: int = 0,
    ) -> Operation:
        """
        Memory Controller node.

        Dot type: ``"MC"``

        Input ports carry load addresses (``*l{n}a``), store addresses
        (``*s{n}a``), store data (``*s{n}d``), and count tokens (``*c{n}``).
        Output ports carry load data (``*l{n}d``) and an end token (``*e``).
        """
        attrs = {
            "bbID": 0,
            "memory": memory,
            "bbcount": bbcount,
            "ldcount": ldcount,
            "stcount": stcount,
        }
        return Operation(name, "mc", operands, out_ports, attrs)

    # ------------------------------------------------------------------
    # CntrlMerge
    # ------------------------------------------------------------------

    @staticmethod
    def cntrl_merge(
        name: str,
        bb_id: int,
        operands: list[Operand],
        out_type: IntType,
        delay: float = 0.366,
        **kw,
    ) -> Operation:
        """
        Control Merge node (N-to-1 selector that also emits the selector index).

        Dot type: ``"CntrlMerge"``

        ``out1`` carries the selected data; ``out2?`` carries the selection index.
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        attrs["delay"] = delay
        return Operation(
            name, "cntrl_merge", operands,
            [Port("out1", out_type), Port("out2", i_opt(1))],
            attrs,
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @staticmethod
    def init(
        name: str,
        bb_id: int,
        data: Operand,
        init_val: Operand,
        out_type: IntType,
        delay: float = 0.366,
        **kw,
    ) -> Operation:
        """
        Init node (selects initial value on first cycle, then passes data).

        Dot type: ``"init Bool false"`` / ``"Init"``
        """
        attrs = _std_tagged_attrs(bb_id, **kw)
        attrs["delay"] = delay
        return Operation(name, "init", [data, init_val], [Port("out1", out_type)], attrs)

    # ------------------------------------------------------------------
    # TaggerUntagger / Tagger / Un_Tagger / Aligner_Branch / Aligner_Mux
    # / Free_Tags_Fifo  (loop-pipelining infrastructure)
    # ------------------------------------------------------------------

    @staticmethod
    def tagger_untagger(
        name: str,
        bb_id: int,
        in1: Operand,
        in2: Operand,
        bw: int,
    ) -> Operation:
        """
        TaggerUntagger node (pre-Graphiti-translation loop primitive).

        Dot type: ``"TaggerUntagger"``
        """
        attrs = _std_tagged_attrs(bb_id)
        return Operation(
            name, "tagger_untagger", [in1, in2],
            [Port("out1", i(bw)), Port("out2", i(bw))],
            attrs,
        )

    @staticmethod
    def tagger(
        name: str,
        bb_id: int,
        data: Operand,
        free_tag: Operand,
        bw: int,
        delay: float = 0.672,
    ) -> Operation:
        """
        Tagger node (attaches a tag to a token).

        Dot type: ``"Tagger"``
        """
        attrs = _std_tagged_attrs(bb_id)
        attrs["delay"] = delay
        return Operation(
            name, "tagger", [data, free_tag],
            [Port("out1", i(bw))],
            attrs,
        )

    @staticmethod
    def un_tagger(
        name: str,
        bb_id: int,
        tagged: Operand,
        bw: int,
    ) -> Operation:
        """
        Un_Tagger node (strips tag from a token, frees the tag slot).

        Dot type: ``"Un_Tagger"``
        """
        attrs = _std_tagged_attrs(bb_id)
        return Operation(
            name, "un_tagger", [tagged],
            [Port("out1", i(bw)), Port("out2", i(bw))],
            attrs,
        )

    @staticmethod
    def aligner_branch(
        name: str,
        bb_id: int,
        data: Operand,
        ctrl: Operand,
        n_out: int,
        bw: int,
        delay: float = 0.672,
    ) -> Operation:
        """
        Aligner_Branch node.

        Dot type: ``"Aligner_Branch"``
        """
        attrs = _std_tagged_attrs(bb_id)
        attrs["delay"] = delay
        out_ports = [Port(f"out{i + 1}", i(bw)) for i in range(n_out)]
        return Operation(name, "aligner_branch", [data, ctrl], out_ports, attrs)

    @staticmethod
    def aligner_mux(
        name: str,
        bb_id: int,
        ctrl: Operand,
        data_inputs: list[Operand],
        bw: int,
        delay: float = 3.637,
    ) -> Operation:
        """
        Aligner_Mux node.

        Dot type: ``"Aligner_Mux"``
        """
        attrs = _std_tagged_attrs(bb_id)
        attrs["delay"] = delay
        return Operation(
            name, "aligner_mux", [ctrl] + data_inputs,
            [Port("out1", i(bw))],
            attrs,
        )

    @staticmethod
    def free_tags_fifo(
        name: str,
        bb_id: int,
        operand: Operand,
    ) -> Operation:
        """
        Free_Tags_Fifo node (recirculates freed tags).

        Dot type: ``"Free_Tags_Fifo"``
        """
        attrs = _std_tagged_attrs(bb_id)
        return Operation(
            name, "free_tags_fifo", [operand],
            [Port("out1", i(32))],
            attrs,
        )

    # ------------------------------------------------------------------
    # Simple I/O (used in test examples without full Dynamatic attributes)
    # ------------------------------------------------------------------

    @staticmethod
    def io(name: str, bw: int = 32) -> Operation:
        """
        Simple I/O port node (test examples).

        Dot type: ``"io"``
        """
        return Operation(name, "io", [], [Port("out1", i(bw))], {})


# ---------------------------------------------------------------------------
# Helper: build an Operand from a Value (source port) to a named destination
# ---------------------------------------------------------------------------

def use(
    value: Value,
    dest_port: str,
    color: Optional[str] = None,
    mem_address: Optional[bool] = None,
    minlen: Optional[int] = None,
) -> Operand:
    """
    Convenience function to create an :class:`Operand`.

    Parameters
    ----------
    value      : Source SSA value (output of some operation).
    dest_port  : Name of the destination port on the consumer node.
    color      : Optional edge color hint.
    mem_address: Optional flag indicating a memory-address edge.
    minlen     : Optional layout hint.
    """
    ea = EdgeAttrs(color=color, mem_address=mem_address, minlen=minlen)
    return Operand(value, dest_port, ea)


# ---------------------------------------------------------------------------
# Example programs
# ---------------------------------------------------------------------------

def example_arithmetic() -> GraphitiModule:
    """
    Reproduce ``tests/arithmetic-example.dot`` programmatically.

    Graph:
        a, b -> add_0 -> fork -> mul_0, mul_1
        c -> mul_0
        d -> mul_1
        mul_0, mul_1 -> add_1 -> mul_2
        e -> mul_2
        mul_2 -> o
    """
    mod = GraphitiModule("arithmetic_example")
    bb = BasicBlock(0, label=None)
    mod.add_block(bb)

    NB = NodeBuilder

    # I/O nodes
    a   = bb.add(NB.io("a"))
    b   = bb.add(NB.io("b"))
    c   = bb.add(NB.io("c"))
    d   = bb.add(NB.io("d"))
    e   = bb.add(NB.io("e"))

    # add_0: a + b
    add_0 = bb.add(NB.operator(
        "add_0", 0, "add_op",
        [use(a.result("out1"), "in1"),
         use(b.result("out1"), "in2")],
        [Port("out2", i(32))],  # port named out2 as in the dot file
    ))

    # fork: replicates add_0.out2 to mul_0 (out2) and mul_1 (out3)
    # Port names match the "from" attributes on edges in the dot file.
    fork = bb.add(NB.fork_ports(
        "fork", 0,
        use(add_0.result("out2"), "in1"),
        [Port("out2", i(32)), Port("out3", i(32))],
    ))

    # mul_0: c * fork.out2
    mul_0 = bb.add(NB.operator(
        "mul_0", 0, "mul_op",
        [use(c.result("out1"), "in1"),
         use(fork.result("out2"), "in2")],
        [Port("out2", i(32))],
    ))

    # mul_1: fork.out3 * d
    mul_1 = bb.add(NB.operator(
        "mul_1", 0, "mul_op",
        [use(fork.result("out3"), "in1"),
         use(d.result("out1"), "in2")],
        [Port("out2", i(32))],
    ))

    # add_1: mul_0 + mul_1
    add_1 = bb.add(NB.operator(
        "add_1", 0, "add_op",
        [use(mul_0.result("out2"), "in1"),
         use(mul_1.result("out2"), "in2")],
        [Port("out2", i(32))],
    ))

    # mul_2: add_1 * e  (out2 is the output connected to 'o')
    mul_2 = bb.add(NB.operator(
        "mul_2", 0, "mul_op",
        [use(add_1.result("out2"), "in1"),
         use(e.result("out1"), "in2")],
        [Port("out2", i(32))],
    ))

    # output o
    o = bb.add(NB.io("o", bw=32))

    # The edge mul_2.out2 -> o is represented by appending o as the sink.
    # In SSA form the connection is already encoded in operands above; we
    # just need to show that mul_2's output feeds 'o'.
    _ = bb.add(Operation(
        "o_connect", "wire",
        [use(mul_2.result("out2"), "in1")],
        [],
        {},
    ))

    return mod


def example_fork_merge() -> GraphitiModule:
    """
    Reproduce ``tests/fork_merge.dot`` programmatically.

    Graph::

        src0 -> fork1 -> fork2 -> merge2 -> snk0
                      +-> merge1
               fork1 -> merge1
               fork2 -> merge2
               merge1 -> merge2
    """
    mod = GraphitiModule("fork_merge")
    bb = BasicBlock(0, label=None)
    mod.add_block(bb)

    NB = NodeBuilder

    src0  = bb.add(NB.io("src0"))
    fork1 = bb.add(NB.fork("fork1", 0, use(src0.result("out1"), "in1"), n_out=2))
    fork2 = bb.add(NB.fork("fork2", 0, use(fork1.result("out2"), "in1"), n_out=2))

    merge1 = bb.add(NB.merge(
        "merge1", 0,
        [use(fork1.result("out1"), "in1"),
         use(fork2.result("out1"), "in2")],
        i(32),
    ))

    merge2 = bb.add(NB.merge(
        "merge2", 0,
        [use(merge1.result("out1"), "in1"),
         use(fork2.result("out2"), "in2")],
        i(32),
    ))

    _snk0 = bb.add(NB.io("snk0"))
    _ = bb.add(Operation(
        "snk0_connect", "wire",
        [use(merge2.result("out1"), "in1")],
        [], {},
    ))

    return mod


def example_dynamatic_no_control_flow() -> GraphitiModule:
    """
    Reproduce ``tests/dynamatic-no-control-flow.dot`` programmatically.

    A simple multiply-then-return circuit in Dynamatic format with one basic
    block, an Entry, several Operators, Forks, a Sink, and an Exit.
    """
    mod = GraphitiModule("dynamatic_no_control_flow")

    # bb1 – the single data-path block
    bb1 = BasicBlock(1, label="block1")
    mod.add_block(bb1)

    # bb0 – global nodes (Exit)
    bb0 = BasicBlock(0, label=None)
    mod.add_block(bb0)

    NB = NodeBuilder

    # In Dynamatic's dot format, output port names on edges use a +1 offset
    # relative to the ``out`` attribute spec on nodes (e.g. spec "out1:32"
    # is referenced as "out2" on edges).  We use the edge port names here as
    # they define the actual dataflow connectivity.

    # ---- block 1 ----
    # Entry: spec out="out1:32", edge from="out2"
    arg     = bb1.add(Operation("arg",     "entry", [],
                                [Port("out2", i(32))],
                                _std_tagged_attrs(1)))
    start_0 = bb1.add(Operation("start_0", "entry", [],
                                [Port("out2", i(0))],
                                {**_std_tagged_attrs(1), "control": True}))

    # Forks (outputs are out2..out{n+1} per edge convention)
    fork_0  = bb1.add(NB.fork_ports("fork_0",  1,
        use(arg.result("out2"), "in1", color="red"),
        [Port("out2", i(32)), Port("out3", i(32)), Port("out4", i(32))]))
    fork_1  = bb1.add(NB.fork_ports("fork_1",  1,
        use(fork_0.result("out4"), "in1", color="red"),
        [Port("out2", i(32)), Port("out3", i(32))]))
    forkC_1 = bb1.add(NB.fork_ports("forkC_1", 1,
        use(start_0.result("out2"), "in1", color="gold3"),
        [Port("out2", i(0)), Port("out3", i(0)), Port("out4", i(0))]))

    # Constants (ctrl token from forkC; output out2 per edge convention)
    cst_0 = bb1.add(Operation("cst_0", "constant",
        [use(forkC_1.result("out2"), "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000002"}))
    cst_1 = bb1.add(Operation("cst_1", "constant",
        [use(forkC_1.result("out3"), "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000002"}))
    cst_2 = bb1.add(Operation("cst_2", "constant",
        [use(forkC_1.result("out4"), "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x0000000A"}))

    # Operators
    mul_0 = bb1.add(NB.operator(
        "mul_0", 1, "mul_op",
        [use(fork_0.result("out2"), "in1", color="red"),
         use(fork_0.result("out3"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=4, II=1,
    ))
    add_1 = bb1.add(NB.operator(
        "add_1", 1, "add_op",
        [use(mul_0.result("out2"), "in1", color="red"),
         use(cst_0.result("out2"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=1.693, latency=0, II=1,
    ))
    shl_2 = bb1.add(NB.operator(
        "shl_2", 1, "shl_op",
        [use(fork_1.result("out2"), "in1", color="red"),
         use(cst_1.result("out2"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=0, II=1,
    ))
    add_3 = bb1.add(NB.operator(
        "add_3", 1, "add_op",
        [use(add_1.result("out2"), "in1", color="red"),
         use(shl_2.result("out2"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=1.693, latency=0, II=1,
    ))
    mul_4 = bb1.add(NB.operator(
        "mul_4", 1, "mul_op",
        [use(add_3.result("out2"), "in1", color="red"),
         use(cst_2.result("out2"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=4, II=1,
    ))
    mul_5 = bb1.add(NB.operator(
        "mul_5", 1, "mul_op",
        [use(mul_4.result("out2"), "in1", color="red"),
         use(fork_1.result("out3"), "in2", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=4, II=1,
    ))
    ret_0 = bb1.add(NB.operator(
        "ret_0", 1, "ret_op",
        [use(mul_5.result("out2"), "in1", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=0, II=1,
    ))

    # ---- block 0 ----
    end_0 = bb0.add(NB.exit(          # noqa: F841
        "end_0", 0,
        [use(ret_0.result("out2"), "in1", color="red")],
        i(32),
    ))

    return mod


def example_dynamatic_if_then_else_merge() -> GraphitiModule:
    """
    Reproduce ``tests/dynamatic-if-then-else-merge.dot`` programmatically.

    An if-then-else circuit in Dynamatic format with three basic blocks, an MC,
    Entry/Exit, and memory load/branch/merge operations.

    Port naming follows the Dynamatic edge convention: output port N in the
    node spec corresponds to ``out{N+1}`` on edges; input port names match.
    """
    mod = GraphitiModule("dynamatic_if_then_else_merge")
    bb1 = BasicBlock(1, label="block1")
    bb2 = BasicBlock(2, label="block2")
    bb3 = BasicBlock(3, label="block3")
    bb0 = BasicBlock(0, label=None)
    for bb in (bb1, bb2, bb3, bb0):
        mod.add_block(bb)

    NB = NodeBuilder

    # Pre-declare values involved in dataflow cycles (MC <-> loads).
    mc_a_out2    = Value("MC_A",    "out2", i_ldata(32, 0))
    mc_a_out3    = Value("MC_A",    "out3", i_ldata(32, 1))
    mc_a_out4    = Value("MC_A",    "out4", i_end(0))
    forkC_1_out2 = Value("forkC_1", "out2", i(0))
    forkC_1_out3 = Value("forkC_1", "out3", i(0))
    forkC_1_out4 = Value("forkC_1", "out4", i(0))
    forkC_1_out5 = Value("forkC_1", "out5", i(0))

    # ---- block 1 ----
    start_0 = bb1.add(Operation("start_0", "entry", [],
        [Port("out2", i(0))],
        {**_std_tagged_attrs(1), "control": True}))

    cst_0 = bb1.add(Operation("cst_0", "constant",
        [use(forkC_1_out2, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000001"}))

    load_1 = bb1.add(NB.operator(
        "load_1", 1, "mc_load_op",
        [use(mc_a_out2,            "in1", color="darkgreen", mem_address=False),
         use(cst_0.result("out2"), "in2", color="red")],
        [Port("out2", i(32)), Port("out3", i_laddr(32, 0))],
        delay=0.0, latency=2, II=1, portId=0, offset=0,
    ))

    fork_2 = bb1.add(NB.fork_ports("fork_2", 1,
        use(load_1.result("out2"), "in1", color="red"),
        [Port("out2", i(32)), Port("out3", i(32))]))

    cst_1 = bb1.add(Operation("cst_1", "constant",
        [use(forkC_1_out3, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000000"}))

    icmp_2 = bb1.add(NB.operator(
        "icmp_2", 1, "icmp_sgt_op",
        [use(fork_2.result("out2"), "in1", color="red"),
         use(cst_1.result("out2"), "in2", color="red")],
        [Port("out2", i(1))],
        delay=1.530, latency=0, II=1,
    ))

    cst_2 = bb1.add(Operation("cst_2", "constant",
        [use(forkC_1_out4, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000000"}))

    fork_0 = bb1.add(NB.fork_ports("fork_0", 1,
        use(icmp_2.result("out2"), "in1", color="red"),
        [Port("out2", i(1)), Port("out3", i(1)), Port("out4", i(1))]))

    forkC_1 = bb1.add(NB.fork_ports("forkC_1", 1,  # noqa: F841
        use(start_0.result("out2"), "in1", color="gold3"),
        [Port("out2", i(0)), Port("out3", i(0)),
         Port("out4", i(0)), Port("out5", i(0))]))

    # Branch nodes: spec out1+(true) / out2-(false) become out2/out3 on edges.
    branch_0 = bb1.add(Operation("branch_0", "branch",
        [use(fork_2.result("out3"),  "in1", color="red"),
         use(fork_0.result("out2"),  "in2", color="red")],
        [Port("out2", i_true(32)), Port("out3", i_false(32))],
        _std_tagged_attrs(1)))

    branch_1 = bb1.add(Operation("branch_1", "branch",
        [use(cst_2.result("out2"),   "in1", color="red"),
         use(fork_0.result("out3"),  "in2", color="red")],
        [Port("out2", i_true(32)), Port("out3", i_false(32))],
        _std_tagged_attrs(1)))

    branchC_2 = bb1.add(Operation("branchC_2", "branch",
        [use(forkC_1_out5,           "in1", color="gold3"),
         use(fork_0.result("out4"),  "in2", color="gold3")],
        [Port("out2", i_true(0)), Port("out3", i_false(0))],
        _std_tagged_attrs(1)))

    # ---- block 2 ----
    cst_3 = bb2.add(Operation("cst_3", "constant",
        [use(branchC_2.result("out3"), "in1", color="gold3", minlen=3)],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(2), "value": "0x00000000"}))

    load_4 = bb2.add(NB.operator(
        "load_4", 2, "mc_load_op",
        [use(mc_a_out3,              "in1", color="darkgreen", mem_address=False),
         use(cst_3.result("out2"),   "in2", color="red")],
        [Port("out2", i(32)), Port("out3", i_laddr(32, 1))],
        delay=0.0, latency=2, II=1, portId=1, offset=0,
    ))

    add_5 = bb2.add(NB.operator(
        "add_5", 2, "add_op",
        [use(branch_0.result("out3"), "in1", color="blue", minlen=3),
         use(load_4.result("out2"),   "in2", color="red")],
        [Port("out2", i(32))],
        delay=1.693, latency=0, II=1,
    ))

    # ---- block 3 ----
    phi_7 = bb3.add(NB.merge(
        "phi_7", 3,
        [use(add_5.result("out2"),    "in1", color="red"),
         use(branch_1.result("out3"), "in2", color="blue", minlen=3)],
        i(32), delay=0.366,
    ))
    phi_7_out2 = Value("phi_7", "out2", i(32))

    ret_0 = bb3.add(NB.operator(
        "ret_0", 3, "ret_op",
        [use(phi_7_out2, "in1", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=0, II=1,
    ))

    # ---- block 0 ----
    _mc = bb0.add(NB.mc(                   # noqa: F841
        "MC_A",
        [use(load_1.result("out3"), "in1", color="darkgreen", mem_address=True),
         use(load_4.result("out3"), "in2", color="darkgreen", mem_address=True)],
        [Port("out2", i_ldata(32, 0)),
         Port("out3", i_ldata(32, 1)),
         Port("out4", i_end(0))],
        memory="A", bbcount=0, ldcount=2, stcount=0,
    ))
    bb0.add(NB.sink("sink_0", 0, use(branch_0.result("out2"), "in1", color="blue", minlen=3)))
    bb0.add(NB.sink("sink_1", 0, use(branch_1.result("out2"), "in1", color="blue", minlen=3)))
    bb0.add(NB.sink("sink_2", 0, use(branchC_2.result("out2"), "in1", color="gold3", minlen=3)))
    bb0.add(NB.exit(
        "end_0", 0,
        [use(mc_a_out4,            "in1", color="gold3"),
         use(ret_0.result("out2"), "in2", color="red")],
        i(32),
    ))

    return mod


def example_dynamatic_if_then_else_mux() -> GraphitiModule:
    """
    Reproduce ``tests/dynamatic-if-then-else-mux.dot`` programmatically.

    Similar to the merge variant but uses a Mux node (``phi_7``) for the phi
    function in block 3.  ``fork_0`` has four outputs (spec out1-out4, edges
    out2-out5); ``out5`` drives the Mux selector.

    Port naming follows the Dynamatic edge convention: output port N in the
    node spec corresponds to ``out{N+1}`` on edges; input port names match.
    """
    mod = GraphitiModule("dynamatic_if_then_else_mux")
    bb1 = BasicBlock(1, label="block1")
    bb2 = BasicBlock(2, label="block2")
    bb3 = BasicBlock(3, label="block3")
    bb0 = BasicBlock(0, label=None)
    for bb in (bb1, bb2, bb3, bb0):
        mod.add_block(bb)

    NB = NodeBuilder

    # Pre-declare values involved in cycles (MC <-> loads).
    mc_a_out2    = Value("MC_A", "out2", i_ldata(32, 0))
    mc_a_out3    = Value("MC_A", "out3", i_ldata(32, 1))
    mc_a_out4    = Value("MC_A", "out4", i_end(0))
    forkC_1_out2 = Value("forkC_1", "out2", i(0))
    forkC_1_out3 = Value("forkC_1", "out3", i(0))
    forkC_1_out4 = Value("forkC_1", "out4", i(0))
    forkC_1_out5 = Value("forkC_1", "out5", i(0))

    # ---- block 1 ----
    start_0 = bb1.add(Operation("start_0", "entry", [],
        [Port("out2", i(0))],
        {**_std_tagged_attrs(1), "control": True}))

    cst_0 = bb1.add(Operation("cst_0", "constant",
        [use(forkC_1_out2, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000001"}))

    load_1 = bb1.add(NB.operator(
        "load_1", 1, "mc_load_op",
        [use(mc_a_out2,            "in1", color="darkgreen", mem_address=False),
         use(cst_0.result("out2"), "in2", color="red")],
        [Port("out2", i(32)), Port("out3", i_laddr(32, 0))],
        delay=0.0, latency=2, II=1, portId=0, offset=0,
    ))

    fork_2 = bb1.add(NB.fork_ports("fork_2", 1,
        use(load_1.result("out2"), "in1", color="red"),
        [Port("out2", i(32)), Port("out3", i(32))]))

    cst_1 = bb1.add(Operation("cst_1", "constant",
        [use(forkC_1_out3, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000000"}))

    icmp_2 = bb1.add(NB.operator(
        "icmp_2", 1, "icmp_sgt_op",
        [use(fork_2.result("out2"), "in1", color="red"),
         use(cst_1.result("out2"), "in2", color="red")],
        [Port("out2", i(1))],
        delay=1.530, latency=0, II=1,
    ))

    cst_2 = bb1.add(Operation("cst_2", "constant",
        [use(forkC_1_out4, "in1", color="gold3")],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(1), "value": "0x00000000"}))

    # fork_0 has 4 spec outputs → 4 edge ports: out2..out5
    fork_0 = bb1.add(NB.fork_ports("fork_0", 1,
        use(icmp_2.result("out2"), "in1", color="red"),
        [Port("out2", i(1)), Port("out3", i(1)),
         Port("out4", i(1)), Port("out5", i(1))]))

    forkC_1 = bb1.add(NB.fork_ports("forkC_1", 1,  # noqa: F841
        use(start_0.result("out2"), "in1", color="gold3"),
        [Port("out2", i(0)), Port("out3", i(0)),
         Port("out4", i(0)), Port("out5", i(0))]))

    # branch_0 data: fork_2.out3; condition: fork_0.out2
    branch_0 = bb1.add(Operation("branch_0", "branch",
        [use(fork_2.result("out3"),  "in1", color="red"),
         use(fork_0.result("out2"),  "in2", color="red")],
        [Port("out2", i_true(32)), Port("out3", i_false(32))],
        _std_tagged_attrs(1)))

    # branch_1 data: cst_2.out2; condition: fork_0.out3
    branch_1 = bb1.add(Operation("branch_1", "branch",
        [use(cst_2.result("out2"),   "in1", color="red"),
         use(fork_0.result("out3"),  "in2", color="red")],
        [Port("out2", i_true(32)), Port("out3", i_false(32))],
        _std_tagged_attrs(1)))

    # branchC_2 data: forkC_1.out5; condition: fork_0.out4
    branchC_2 = bb1.add(Operation("branchC_2", "branch",
        [use(forkC_1_out5,           "in1", color="gold3"),
         use(fork_0.result("out4"),  "in2", color="gold3")],
        [Port("out2", i_true(0)), Port("out3", i_false(0))],
        _std_tagged_attrs(1)))

    # ---- block 2 ----
    cst_3 = bb2.add(Operation("cst_3", "constant",
        [use(branchC_2.result("out3"), "in1", color="gold3", minlen=3)],
        [Port("out2", i(32))],
        {**_std_tagged_attrs(2), "value": "0x00000000"}))

    load_4 = bb2.add(NB.operator(
        "load_4", 2, "mc_load_op",
        [use(mc_a_out3,              "in1", color="darkgreen", mem_address=False),
         use(cst_3.result("out2"),   "in2", color="red")],
        [Port("out2", i(32)), Port("out3", i_laddr(32, 1))],
        delay=0.0, latency=2, II=1, portId=1, offset=0,
    ))

    add_5 = bb2.add(NB.operator(
        "add_5", 2, "add_op",
        [use(branch_0.result("out3"), "in1", color="blue", minlen=3),
         use(load_4.result("out2"),   "in2", color="red")],
        [Port("out2", i(32))],
        delay=1.693, latency=0, II=1,
    ))

    # ---- block 3 ----
    # phi_7 is a Mux: selector = fork_0.out5 (in1?), data: add_5.out2 (in2), branch_1.out3 (in3)
    # Dynamatic edge convention: spec out1 → edge out2
    _phi7_attrs = _std_tagged_attrs(3)
    _phi7_attrs["delay"] = 0.366
    phi_7 = bb3.add(Operation("phi_7", "mux",
        [use(fork_0.result("out5"),    "in1", color="red"),
         use(add_5.result("out2"),     "in2", color="red"),
         use(branch_1.result("out3"),  "in3", color="blue", minlen=3)],
        [Port("out2", i(32))],
        _phi7_attrs))

    ret_0 = bb3.add(NB.operator(
        "ret_0", 3, "ret_op",
        [use(phi_7.result("out2"), "in1", color="red")],
        [Port("out2", i(32))],
        delay=0.0, latency=0, II=1,
    ))

    # ---- block 0 ----
    bb0.add(NB.mc(
        "MC_A",
        [use(load_1.result("out3"), "in1", color="darkgreen", mem_address=True),
         use(load_4.result("out3"), "in2", color="darkgreen", mem_address=True)],
        [Port("out2", i_ldata(32, 0)),
         Port("out3", i_ldata(32, 1)),
         Port("out4", i_end(0))],
        memory="A", bbcount=0, ldcount=2, stcount=0,
    ))
    bb0.add(NB.sink("sink_0", 0, use(branch_0.result("out2"), "in1", color="blue", minlen=3)))
    bb0.add(NB.sink("sink_1", 0, use(branch_1.result("out2"), "in1", color="blue", minlen=3)))
    bb0.add(NB.sink("sink_2", 0, use(branchC_2.result("out2"), "in1", color="gold3", minlen=3)))
    bb0.add(NB.exit(
        "end_0", 0,
        [use(mc_a_out4,            "in1", color="gold3"),
         use(ret_0.result("out2"), "in2", color="red")],
        i(32),
    ))

    return mod



_EXAMPLES: dict[str, tuple[str, "Callable[[], GraphitiModule]"]] = {
    "arithmetic":            ("tests/arithmetic-example.dot",             example_arithmetic),
    "fork_merge":            ("tests/fork_merge.dot",                     example_fork_merge),
    "no_control_flow":       ("tests/dynamatic-no-control-flow.dot",      example_dynamatic_no_control_flow),
    "if_then_else_merge":    ("tests/dynamatic-if-then-else-merge.dot",   example_dynamatic_if_then_else_merge),
    "if_then_else_mux":      ("tests/dynamatic-if-then-else-mux.dot",     example_dynamatic_if_then_else_mux),
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Emit Graphiti IR (MLIR-like) for built-in example programs.  "
            "No dot files are read at runtime; all graphs are constructed "
            "programmatically from their dot-file descriptions."
        )
    )
    parser.add_argument(
        "example",
        nargs="?",
        choices=list(_EXAMPLES),
        default=None,
        help="Name of the example to emit (default: emit all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available examples (corresponds to dot file):")
        for name, (dot_file, _) in _EXAMPLES.items():
            print(f"  {name:20s}  {dot_file}")
        return

    to_emit = [args.example] if args.example else list(_EXAMPLES)
    for name in to_emit:
        dot_file, builder = _EXAMPLES[name]
        print(f"// ---- {name}  (from {dot_file}) ----")
        builder().print()
        print()


if __name__ == "__main__":
    main()
