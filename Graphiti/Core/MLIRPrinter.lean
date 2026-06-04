/-
Copyright (c) 2024 VCA Lab, EPFL. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yann Herklotz
-/

module

public import Graphiti.Core.DynamaticPrinter

public section

open Batteries (AssocList)

namespace Graphiti

/--
Format a single attribute value for the MLIR-like IR.

Attribute keys whose values are numeric or boolean in the DOT source
(bbID, bbcount, ldcount, stcount, II, latency, delay, tagger_id,
taggers_num, tagged, offset, portId) are emitted as bare tokens.
All other attribute values are wrapped in double-quotes with any inner
double-quotes backslash-escaped.
-/
def formatMLIRAttrVal (key val : String) : String :=
  let isUnquoted :=
    key = "bbID" || key = "bbcount" || key = "ldcount" || key = "stcount"
    || key = "II" || key = "latency" || key = "delay"
    || key = "tagger_id" || key = "taggers_num" || key = "tagged"
    || key = "offset" || key = "portId"
  if isUnquoted then val
  else s!"\"{val.replace "\"" "\\\""}\""

/--
Format a list of extra node attributes as a comma-separated MLIR fragment.

`isMC` controls whether `in`/`out` port-spec attributes are included: for
non-MC nodes the port specs are inferred from the type system and emitted
separately, so they are skipped here (as in `DynamaticPrinter.formatOptions`).

Returns a string of the form `, key1 = val1, key2 = val2, ...` (with a
leading ", " before the first attribute), or the empty string when all
entries are skipped.
-/
def formatMLIRAttrs (isMC : Bool) : List (String × String) → String
  | [] => ""
  | x :: l =>
    let fmtPair (sl sr : String) : String :=
      -- `in` port specs may have an internal 'p' prefix that must be stripped
      let v := if sl = "in" then removeLetter 'p' sr else sr
      s!"{sl} = {formatMLIRAttrVal sl v}"
    let skipInOut (sl : String) : Bool := (sl = "in" || sl = "out") && !isMC
    let first :=
      if skipInOut x.1 then ""
      else ", " ++ fmtPair x.1 x.2
    l.foldl
      (λ s (sl, sr) =>
        if skipInOut sl then s
        else s ++ ", " ++ fmtPair sl sr)
      first

/--
Emit the rewritten graph as a custom MLIR-like intermediate representation.

Format:
  // graphiti-ir v1.0
  module {
    graphiti.circuit {
      %"node_id" = graphiti.node {type = "T", attr1 = val1, in = "...", out = "..."}
      ...
      graphiti.connect %"src_id"#out_port -> %"dst_id"#in_port
      ...
    }
  }

This replaces the DOT intermediate format that was previously written to a
temporary file and consumed by `graphiti-to-dynamatic.py`.  The Python
consumer now calls `graphiti_ir.parse_mlir` instead of
`graphiti_conv.parse_dot`, giving it a format that is closer to MLIR and
makes no use of DOT files as Python inputs.
-/
def mlirString (a : ExprHigh String (String × Nat)) (t : TypeUF)
    (m : AssocList String (AssocList String String)) : Except String String := do
  let a ← ofOption' "could not normalise names" a.normaliseNames
  let modules ←
    a.modules.foldlM
      (λ s k v => do
        let typeName := graphitiToDynamatic v.2.1 |>.1
        match m.find? k with
        | some input_fmt =>
          let shouldNotInfer := v.2.1 = "mc" || graphitiPrefix.isPrefixOf v.2.1
          let typs ←
            if shouldNotInfer then pure (∅, ∅)
            else inferTypeInPortMapping t v.1.canonPortMapping v.2
          let formatInOut :=
            if shouldNotInfer then ""
            else s!", in = \"{toPortList typs.1}\", out = \"{toPortList typs.2}\""
          return s ++
            s!"    %\"{k}\" = graphiti.node \{type = \"{typeName}\"{formatMLIRAttrs shouldNotInfer input_fmt.toList}{formatInOut}}\n"
        | none =>
          let typs ← inferTypeInPortMapping t v.1.canonPortMapping v.2
          return s ++
            s!"    %\"n_{k}\" = graphiti.node \{type = \"{typeName}\", in = \"{toPortList typs.1}\", out = \"{toPortList typs.2}\"}\n"
      ) ""
  let frmat (i : InstIdent String) :=
    if m.contains (toString i) then toString i else "n_" ++ toString i
  let connections :=
    a.connections.foldl
      (λ s => λ | ⟨oport, iport⟩ =>
        s ++
        s!"    graphiti.connect %\"{frmat oport.inst}\"#{oport.name}"
          ++ s!" -> %\"{frmat iport.inst}\"#{removeLetter 'p' iport.name}\n") ""
  .ok s!"// graphiti-ir v1.0\nmodule \{\n  graphiti.circuit \{\n{modules}{connections}  }\n}\n"

end Graphiti
