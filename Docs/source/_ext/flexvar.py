"""
flexvar - A Sphinx domain for documenting variables with flexible names.

Supports variable names containing characters like <, >, /, commas, etc.
Provides type annotation and default value support, styled like the Python domain.

Usage
-----

Directive::

    .. fv:var:: my/variable<T>

        :type: list<int>
        :default: []

        Description of the variable.

Role::

    See :fv:var:`my/variable<T>` for details.

    # With explicit title (whitespace required before the `<`):
    See :fv:var:`My Var <my/variable<T>>` for details.

    # With backslash-escaped angle brackets (same as the option role):
    See :fv:var:`my/variable\<T\>` for details.

    # With inline value (value shown in link text, stripped for lookup):
    See :fv:var:`my/variable<T> = []` for details.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, List, cast

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.roles import XRefRole, ws_re
from sphinx.util.docfields import Field
from sphinx.util.nodes import make_id, make_refnode


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------

class FlexVarDirective(ObjectDescription[str]):
    """
    Directive: .. fv:var:: <name>

    Options
    -------
    :type: <type string>   (optional) the variable's type
    :default: <value>      (optional) the variable's default value
    :noindex:              suppress index entry
    """

    option_spec = {
        "type": directives.unchanged,
        "default": directives.unchanged,
        "noindex": directives.flag,
    }

    # Disallow multiple names on a single directive line (commas in the name
    # would be mis-parsed otherwise).
    allow_nesting = False

    # ------------------------------------------------------------------
    # Signature parsing / rendering
    # ------------------------------------------------------------------

    def _parse_inline(self, text: str) -> list[nodes.Node]:
        """Parse *text* as RST inline content and return the resulting nodes."""
        nodes_, _ = self.state.inline_text(text, self.lineno)
        return nodes_

    def handle_signature(self, sig: str, signode: addnodes.desc_signature) -> str:
        """Build the rendered signature node and return the canonical name."""
        name = sig.strip()

        signode["fullname"] = name
        signode["ids"] = []  # filled in add_target_and_index

        # The variable name itself
        signode += addnodes.desc_name(name, name)

        # Optional type annotation  `: <type>`
        typ = self.options.get("type", "")
        if typ:
            signode += addnodes.desc_annotation(
                typ, '',
                addnodes.desc_sig_punctuation('', ':'),
                addnodes.desc_sig_space(),
                *self._parse_inline(typ),
            )

        # Optional default value  ` = <value>`
        value = self.options.get("default", "").strip()
        if value:
            signode += addnodes.desc_annotation(
                value, '',
                addnodes.desc_sig_space(),
                addnodes.desc_sig_punctuation('', '='),
                addnodes.desc_sig_space(),
                *self._parse_inline(value),
            )

        return name

    # ------------------------------------------------------------------
    # Index + target registration
    # ------------------------------------------------------------------

    def add_target_and_index(
        self, name: str, sig: str, signode: addnodes.desc_signature
    ) -> None:
        node_id = make_id(self.env, self.state.document, '', name)
        signode["ids"].append(node_id)
        self.state.document.note_explicit_target(signode)

        domain = cast(FlexVarDomain, self.env.get_domain("fv"))
        domain.note_var(
            name=name,
            docname=self.env.docname,
            node_id=node_id,
            type_str=self.options.get("type", ""),
            default_str=self.options.get("default", ""),
        )

        if "noindex" not in self.options:
            self.indexnode["entries"].append(
                ("single", name + " (variable)", node_id, "", None)
            )


# ---------------------------------------------------------------------------
# XRef Role
# ---------------------------------------------------------------------------

class FlexVarRole(XRefRole):
    """
    Role: :fv:var:`name` or :fv:var:`Title <name>`

    Two customisations over the base ``XRefRole``:

    1. **Generic-style names** — variable names may contain ``<`` and ``>``
       (e.g. ``filter<T>``).  We require whitespace before the ``<`` that
       separates an explicit title from its target, so bare names like
       ``filter<T>`` are never mis-split.  This is done by overriding
       ``explicit_title_re``, which ``ReferenceRole.__call__`` uses directly.

    2. **Inline value syntax** — a cross-reference may include a value
       expression after `` = `` (e.g. ``:fv:var:`timeout = 30```).  The value
       is kept in the displayed title but stripped from the lookup target so
       that it still resolves to the ``.. fv:var:: timeout`` entry.
       Backslash-escape support (``\<``, ``\>``) comes for free from the
       base class.
    """

    # Same as ReferenceRole.explicit_title_re but with \s+ instead of \s*,
    # so whitespace before the `<` is required for explicit-title syntax.
    # \x00 means the "<" was backslash-escaped — preserve that lookbehind.
    explicit_title_re = re.compile(r'^(.+?)\s+(?<!\x00)<(.*?)>$', re.DOTALL)

    # Matches an inline value expression: "varname = value" or "varname[=value]"
    # The name portion (before = or [=) is captured as group 1.
    _value_re = re.compile(r'^(.+?)(?:\s*=\s*.*|\[=.*\])$', re.DOTALL)

    def process_link(
        self,
        env: BuildEnvironment,
        refnode: nodes.Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        """Strip any inline value expression from the target, keeping it in the title."""
        if not has_explicit_title:
            m = self._value_re.match(target)
            if m:
                target = m.group(1).strip()
        return title, ws_re.sub(' ', target)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class FlexVarDomain(Domain):
    """The ``fv`` domain for flexible variable documentation."""

    name = "fv"
    label = "FlexVar"

    object_types = {
        "var": ObjType("variable", "var"),
    }

    directives = {
        "var": FlexVarDirective,
    }

    roles = {
        "var": FlexVarRole(),
    }

    initial_data: dict = {
        "vars": {},  # name -> {"docname": str, "node_id": str, "type": str, "default": str}
    }

    @property
    def vars(self) -> dict:
        return self.data.setdefault("vars", {})

    def note_var(
        self,
        name: str,
        docname: str,
        node_id: str,
        type_str: str = "",
        default_str: str = "",
    ) -> None:
        self.vars[name] = {
            "docname": docname,
            "node_id": node_id,
            "type": type_str,
            "default": default_str,
        }

    def clear_doc(self, docname: str) -> None:
        to_remove = [k for k, v in self.vars.items() if v["docname"] == docname]
        for k in to_remove:
            del self.vars[k]

    def merge_domaindata(self, docnames: List[str], otherdata: dict) -> None:
        for name, info in otherdata.get("vars", {}).items():
            if info["docname"] in docnames:
                self.vars[name] = info

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Any,
        typ: str,
        target: str,
        node: addnodes.pending_xref,
        contnode: nodes.Element,
    ) -> nodes.Element | None:
        info = self.vars.get(target)
        if info is None:
            return None
        return make_refnode(
            builder,
            fromdocname,
            info["docname"],
            info["node_id"],
            contnode,
            target,
        )

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        for name, info in self.vars.items():
            yield (
                name,             # name
                name,             # dispname
                "var",            # type
                info["docname"],  # docname
                info["node_id"],  # anchor
                1,                # priority
            )


# ---------------------------------------------------------------------------
# Extension entry point
# ---------------------------------------------------------------------------

def setup(app: Sphinx) -> dict:
    app.add_domain(FlexVarDomain)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
