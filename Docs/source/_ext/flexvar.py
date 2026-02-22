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
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple, cast

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.roles import XRefRole
from sphinx.util.docfields import Field, TypedField
from sphinx.util.nodes import make_refnode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(name: str) -> str:
    """Turn an arbitrary variable name into a valid HTML id."""
    # Replace chars that are awkward in IDs with underscores
    return re.sub(r"[^\w.\-]", "_", name)


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------

class FlexVarDirective(ObjectDescription):
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

    def handle_signature(self, sig: str, signode: addnodes.desc_signature) -> str:
        """Build the rendered signature node and return the canonical name."""
        name = sig.strip()

        signode["fullname"] = name
        signode["ids"] = []  # filled in add_target_and_index

        # "var " prefix, styled like py:data
        signode += addnodes.desc_annotation("var ", "var ")

        # The variable name itself
        signode += addnodes.desc_name(name, name)

        # Optional type annotation  `: <type>`
        type_str = self.options.get("type", "").strip()
        if type_str:
            signode += addnodes.desc_sig_punctuation("", " : ")
            signode += addnodes.desc_sig_name("", type_str)

        # Optional default value  ` = <value>`
        default_str = self.options.get("default", "").strip()
        if default_str:
            signode += addnodes.desc_sig_punctuation("", " = ")
            signode += nodes.literal("", default_str)

        return name

    # ------------------------------------------------------------------
    # Index + target registration
    # ------------------------------------------------------------------

    def add_target_and_index(
        self, name: str, sig: str, signode: addnodes.desc_signature
    ) -> None:
        node_id = "fv.var." + _make_id(name)

        # Avoid duplicate IDs
        if node_id not in self.state.document.ids:
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

    # ------------------------------------------------------------------
    # doc-field-types exposed to the body (kept minimal)
    # ------------------------------------------------------------------
    doc_field_types = [
        Field("type", label="Type", has_arg=False, names=("type",)),
        Field("default", label="Default", has_arg=False, names=("default",)),
    ]


# ---------------------------------------------------------------------------
# XRef Role
# ---------------------------------------------------------------------------

class FlexVarRole(XRefRole):
    """
    Role: :fv:var:`name` or :fv:var:`Title <n>`

    Variable names may contain ``<`` and ``>`` (e.g. ``filter<T>``).
    Sphinx's ``split_explicit_title`` and ``utils.unescape()`` in the base
    ``XRefRole.__call__`` already handle backslash-escaped angle brackets
    (``\<``, ``\>``) correctly — the same way the built-in ``option`` role
    does — so we don't need to touch ``__call__`` at all.

    The only thing we override is ``process_link``, to redo the title/target
    split with a stricter heuristic: an explicit title is only recognised when
    there is **whitespace before the separating** ``<``, so bare generic-style
    names like ``filter<T>`` are never mis-split.
    """

    # Matches an explicit-title reference: `Some Title <actual/target<T>>`
    # Requires whitespace before the opening `<` so that bare names like
    # `filter<T>` are never treated as title + target.
    _explicit_title_re = re.compile(r"^(.+?)\s+<(.+)>\s*$", re.DOTALL)

    def process_link(
        self,
        env: BuildEnvironment,
        refnode: nodes.Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        """
        Re-split title and target using our stricter heuristic.

        By the time this is called, ``XRefRole.__call__`` has already run
        ``utils.unescape()`` on both ``title`` and ``target``, so escaped
        characters like ``\<`` have been resolved to literal ``<``.  We
        just need to re-apply our own splitting logic on the full unescaped
        string (which is ``title`` when no explicit title was detected by
        Sphinx, or the concatenation when one was).
        """
        # Reconstruct the full unescaped content string.  When Sphinx's own
        # split_explicit_title found an explicit title, title and target are
        # already separate; when it didn't, title == target == the whole text.
        # Either way, re-running our regex on title (which equals the full
        # string in the no-explicit-title case) is correct.
        full = title if not has_explicit_title else f"{title} <{target}>"
        m = self._explicit_title_re.match(full)
        if m:
            return m.group(1), m.group(2)
        # No explicit title — the whole string is both title and target.
        return title, title


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

    # Stored per-document: name -> (docname, node_id, type, default)
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

    def get_objects(self):
        for name, info in self.vars.items():
            yield (
                name,                # name
                name,                # dispname
                "var",               # type
                info["docname"],     # docname
                info["node_id"],     # anchor
                1,                   # priority
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
