"""
Convert WarpX parameters.rst to use fv:var directives.

Only top-level param bullets are converted; nested param bullets inside
description bodies are left as raw RST (no recursive conversion).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple, Any


# ── Regex ─────────────────────────────────────────────────────────────────────

# Matches a RST bullet that introduces a parameter:
#   group 1 – leading whitespace (indent depth)
#   group 2 – bullet character (* or -)
#   group 3 – parameter name (inside double backticks)
#   group 4 – remainder of line after the closing ``
PARAM_BULLET_RE = re.compile(r'^( *)([*\-]) ``([^`]+)``(.*)', re.DOTALL)


# Span
class Span(NamedTuple):
    """Half-open line range [start, end) within the source file."""
    start: int
    end: int

# Directive
class Directive:
    """
    All information extracted from one parameter bullet.
    """

    # ── Directive construction ────────────────────────────────────────────────────
    def __init__(self, lines: list[str], span: Span | None = None):
        """Construct a Directive from the source lines covered by span."""
        if span is None:
            span = Span(0, len(lines))

        bullet_line = lines[span.start]
        annotation = BulletAnnotation(bullet_line)

        first_name: str = annotation.first_name
        bullet_indent: int = annotation.bullet_indent

        names = merge_multiple_names([first_name] + annotation.extra_names)

        # Raw body: every source line after the bullet, newlines stripped
        raw_body: list[str] = [lines[k].rstrip('\n') for k in range(span.start + 1, span.end)]

        # Find body indent
        body_indent = detect_body_indent(raw_body, bullet_indent=bullet_indent)

        body: list[str] = []
        # if annotation.inline_desc:
        #     body.append(annotation.inline_desc + " inline_desc_TEST")
        for raw in raw_body:
            if raw.strip() == '':
                body.append('')
            else:
                col = len(raw) - len(raw.lstrip())
                if col < body_indent:
                    print(f"\nWARNING: {names}, bullet_indent={bullet_indent}, col={col}, body_indent={body_indent}, raw={raw}")
                body.append(raw[min(body_indent, col):])

        self.bullet_line: str = bullet_line
        self.names: list[str] = names

        self.raw_name_list: list[str] = [first_name] + annotation.extra_names

        self.type_str: str = annotation.type_str
        self.default_str: str = annotation.default_str
        self.unit_str: str = annotation.unit_str
        self.comment_str: str = annotation.comment_str

        self.annotation: BulletAnnotation = annotation

        self.raw_body: list[str] = raw_body
        self.body: list[str] = body
        self.bullet_indent: int = bullet_indent
        self.body_indent: int = body_indent

        self.source_span: Span = span

    @property
    def name(self) -> str:
        return self.names[0]

    # RST output rendering
    def render(self) -> list[str]:
        """
        Render a Directive to output lines.

        One ``.. fv:var::`` block is emitted per name in d.names, all sharing
        the same type, default, and body.  Blocks are separated by blank lines.
        """
        bullet_indent: str = ' ' * self.bullet_indent
        body_indent: str = ' ' * self.body_indent
        out: list[str] = []

        l_use_raw_annotation = False

        l_past_first_name = False
        for name in self.names[:1]:
            if l_past_first_name:
                out.append('')
            l_past_first_name = True
            out.append(f'{bullet_indent}.. fv:var:: {name}')

            if len(self.raw_name_list) > 1 or self.raw_name_list[0] != name:
                other_names = [ elem.replace(" ", "") for elem in self.raw_name_list ]
                other_names_str = " ".join(other_names)
                out.append(f"{body_indent}:othernames: {other_names_str}")
            if self.type_str:
                out.append(f'{body_indent}:type: {self.type_str}')
            if self.unit_str:
                out.append(f'{body_indent}:unit: {self.unit_str}')
            if self.default_str:
                out.append(f'{body_indent}:default: {self.default_str}')
            if self.annotation.optional_flag:
                out.append(f'{body_indent}:optional:')
            if self.comment_str:
                out.append(f'{body_indent}:comment: {self.comment_str}')
            if l_use_raw_annotation:
                raw_anno_txt = self.annotation.raw_annotation.strip().lstrip('.:').strip()
                out.append(f"{body_indent}:annotation: {raw_anno_txt}")

            body_lines = [ f'{body_indent}{line}'.rstrip() for line in self.body ]
            if len(body_lines) < 1 or body_lines[0] != "":
                body_lines.insert(0, "")
            out.extend(body_lines)

        # while out and out[0].strip == '':
        #     out.pop(0)
        # while out and out[-1].strip() == '':
        #     out.pop()

        return out

# Type normalisation
WORD_NORMS: dict[str, str] = {
    'integer': 'int',
    'integers': 'int',
    'boolean': 'bool',
    'double': 'float',
    'doubles': 'float',
    'string': 'str',
}

def normalise_type(s: str) -> str:
    """Normalise a type string extracted from a parameter bullet."""
    # if not s:
    #     return s
    # s = s.strip()
    # # ``0`` or ``1`` → bool
    # s = re.sub(r'[`]*0[`]*\s+or\s+[`]*1[`]*', 'bool', s)
    # # Normalise standalone type words
    # for old, new in WORD_NORMS.items():
    #     s = re.sub(rf'(?<!\w){re.escape(old)}(?!\w)', new, s, flags=re.IGNORECASE)
    # # Fix double-backtick artifacts: ``bool`` → `bool`
    # s = re.sub(r'``(int|float|str|bool)``', r'`\1`', s)
    return s

# Parenthesis finder
def find_paren_end(s: str) -> int:
    """Return the index of the ')' matching the '(' assumed to be at s[0].
    Skips ``double-backtick`` spans to avoid false matches inside code."""
    depth = 0
    i = 0
    while i < len(s):
        if s[i:i+2] == '``':
            i += 2
            end = s.find('``', i)
            i = end + 2 if end != -1 else len(s)
            continue
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s) - 1

# ── Bullet annotation ─────────────────────────────────────────────────────────

class BulletAnnotation:
    """Parse bullet line."""

    def __init__(self, bullet_line: str):
        """Parse bullet line.

        Handles the following source patterns (non-exhaustive):
            , ``co.name`` and ``other.name`` (`type`; default: X) desc…
            (`type`; default: X) optional
            optional (default: X) desc…
            (`string`: e.g. ``...``)
            (``0`` or ``1``; default is ``1`` for true)
            plain description with no annotation
        """
        unit_str: str = ""
        type_str: str = ""
        default_str: str = ""
        comment_str: str = ""
        optional_flag: bool = False

        m_bullet = PARAM_BULLET_RE.match(bullet_line)
        assert m_bullet is not None

        self.bullet_indent: int = len(m_bullet.group(1))
        self.first_name: str = m_bullet.group(3).strip()

        rest = m_bullet.group(4).rstrip('\n')

        s = rest.strip()

        # ── Collect additional co-listed names ────────────────────────────────────
        extra_names: list[str] = []
        while True:
            m = re.match(r'^(?:,\s*|(?:and|&)\s+|,\s*(?:and|&)\s+)``([^`]+)``\s*(.*)', s, re.DOTALL)
            if m:
                extra_names.append(m.group(1).strip())
                s = m.group(2).strip()
            else:
                break
        raw_annotation = s

        # Units
        if "dimensionless" in s:
            # unit_str = "dimensionless"
            comment_str += "dimensionless"
            s = s.replace("dimensionless", "")
        elif "[meter]" in s:
            unit_str = "meters"
            s = s.replace("[meter]", "")
        else:
            m = re.match(r'\((?:.*)([,;]\s*in\s+)([^\);]+)[\);]', s, re.DOTALL)
            if m:
                unit_str = m.group(2)
                s = s.replace(m.group(1)+m.group(2), "")

        # Optional flag
        if "optional" in s:
            optional_flag = True

        if optional_flag:
            # Remove "optional" from s
            s = re.sub(r'optional', '', s, flags=re.IGNORECASE)

        # Type, default, comments
        if s.startswith('('):
            # Main parenthesised annotation
            end = find_paren_end(s)
            assert s[end] == ')'
            first_paren_txt = s[1:end]
            after = s[end + 1:].strip()
            # print(f"\nfirst_name: {self.first_name}\n  s[1:end]: {s[1:end]}\n  s[end]: {s[end]}\n")

            first_paren_txt = first_paren_txt.strip()

            # "; default …" or ", default …"
            m_default_is = re.search(
                r'[,;\s]*default(?:\s+is)?:?(?:\s*\=)?\s*([^\(\)]+)',
                first_paren_txt, re.IGNORECASE
            )
            # ";… X by default"
            m_by_default = re.search(
                r'[,;]\s*(\S+)\s+by\s+default$',
                first_paren_txt, re.IGNORECASE
            )

            if m_default_is:
                type_str = first_paren_txt[:m_default_is.start()].strip()
                default_str = m_default_is.group(1).strip().rstrip(')')
            elif m_by_default:
                # print(f"\nm2 match, s: '{s}'\n")
                type_str = first_paren_txt[:m_by_default.start()].strip()
                default_str = m_by_default.group(1).strip()
            else:
                type_str = first_paren_txt
                default_str = ""

            # "type: comment"
            m_type_comment = re.search(
                r'(.+)[:](.+)',
                type_str, re.DOTALL
            )
            if m_type_comment:
                type_str = m_type_comment.group(1)
                comment_str += m_type_comment.group(2)

            # Strip a bare "optional" that sometimes follows
            after = re.sub(r'^optional\b', '', after, flags=re.IGNORECASE).strip()
            # A second parenthesis may carry the default when the first didn't
            if after.startswith('(') and not default_str:
                end2 = find_paren_end(after)
                dm = re.match(r'default:?\s*(.*)', after[1:end2].strip(), re.IGNORECASE)
                if dm:
                    default_str = dm.group(1).strip()
                comment_str += after[end2 + 1:].strip()
            else:
                comment_str += after

            # if type_str and not default_str and after:
            #     print("")
            #     print(f"first_name: {self.first_name}")
            #     print(f"  bullet_line: {bullet_line}")
            #     print(f"  type_str: {type_str}")
            #     print(f"  default_str: {default_str}")
            #     print(f"  after: {after}")
            #     print("")

        elif s:
            # print(f"\nCASE 3, first_name: '{self.first_name}', s: '{s}'\n")
            comment_str = s

        # Strip errant punctuation
        type_str = type_str.strip(".:;,= ")
        default_str = default_str.strip(":;,= ")
        unit_str = unit_str.strip(".:;,= ")
        comment_str = comment_str.lstrip(".;").strip(":,= ")

        # co-listed names, e.g. from "and ``foo.hi``"
        self.extra_names: list[str]  = extra_names

        self.type_str: str = type_str
        self.default_str: str = default_str
        self.unit_str: str = unit_str
        self.optional_flag: bool = optional_flag
        # descriptive text on the bullet line itself
        self.comment_str: str = comment_str

        self.raw_bullet_line: str = bullet_line
        self.raw_rest: str = rest
        self.raw_annotation: str = raw_annotation


# ── multiple name merging ────────────────────────────────────────────────────────

def _merge_multiple_names(names: list[str]) -> list[str]:
    """Collapse lo/hi name pairs into a single combined name.

    Three patterns are handled:
      1. Simple pair:       foo_lo + foo_hi              → foo_lo/hi
      2. Suffixed pair:     foo_lo_x/y/z + foo_hi_x/y/z → foo_lo/hi_x/y/z
      3. Multi-axis pair:   at_xlo/ylo/zlo + at_xhi/yhi/zhi
                             → at_xlo/ylo/zlo/xhi/yhi/zhi

    Names that do not participate in any merge are returned unchanged.
    When three names are present and two merge, the merged name and the
    unmerged name are both returned (e.g. [merged, at_eb]).
    """
    if len(names) < 2:
        return names

    result: list[str] = []
    used: set[int] = set()

    # First try spliting by `_` and see if only

    patterns_pre_mid_suf = [
        # Pattern: <PREFIX>(nx|ny|nz|nox|noy|noz)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>nx|ny|nz|nox|noy|noz)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(xlo/ylo/zlo|xhi/yhi/zhi|at_eb)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>xlo/ylo/zlo|xhi/yhi/zhi|eb)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(sigma|epsilon|mu|field|particle)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>sigma|epsilon|mu|field|particle)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(lo|hi)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>lo|hi)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(lo|hi)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>Ex|Ey|Ez|Bx|By|Bz)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(x|y|z)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>x|y|z)(?P<suffix>.*)$'),
        # Pattern: <PREFIX>(E|B)<SUFFIX>
        re.compile(r'^(?P<prefix>.*)(?P<middle>E|B)(?P<suffix>.*)$'),
    ]

    for pattern in patterns_pre_mid_suf:
        matchlist: list[re.Match] = []
        for name in names:
            m = re.match(pattern, name)
            if m:
                matchlist.append(m)
        if len(matchlist) != len(names):
            continue
        # print(f"\nmerge_multiple_names: found match\n  names = {names}\n  pattern = {pattern}")

        pre_list: list[str] = [m.group('prefix') for m in matchlist]
        mid_list: list[str] = [m.group('middle') for m in matchlist]
        suf_list: list[str] = [m.group('suffix') for m in matchlist]

        if len(set(pre_list)) != 1:
            # print(f"  pre_list = {pre_list}")
            continue
        if len(set(suf_list)) != 1:
            # print(f"  suf_list = {suf_list}")
            continue
        if len(set(mid_list)) != len(mid_list):
            # print(f"  mid_list = {mid_list}")
            continue

        result.append("".join([
            pre_list[0],
            # '<',
            '/'.join(mid_list),
            # '>',
            suf_list[0]
        ]))

        # print(f"\nmerge_multiple_names: found match\n  names  = {names}\n  pattern = {pattern.pattern}\n  result={result}")
        return result

    # Pattern: <PREFIX>xmin,ymin,zmin & <PREFIX>xmax,ymax,zmax
    if len(names) == 2:
        matchlist: list[re.Match] = []
        pattern = re.compile(r'^(.*)(xmin,ymin,zmin|xmax,ymax,zmax)$')
        for name in names:
            m = re.match(pattern, name)
            if m:
                matchlist.append(m)
        pre_list: list[str] = [m.group(1) for m in matchlist]
        mid_list: list[str] = [m.group(2) for m in matchlist]

        if len(matchlist) == len(names):
            result.append(pre_list[0] + ','.join(mid_list))
            # print(f"\nmerge_multiple_names: found match\n  names  = {names}\n  pattern = {pattern.pattern}\n  result={result}")
            return result

    for i, name in enumerate(names):
        if i in used:
            continue
        merged = False
        for j, other in enumerate(names):
            if j <= i or j in used:
                continue
            # Pattern 1: simple _lo / _hi
            m_lo = re.match(r'^(.*?)_lo$', name)
            m_hi = re.match(r'^(.*?)_hi$', other)
            if m_lo and m_hi and m_lo.group(1) == m_hi.group(1):
                result.append(m_lo.group(1) + '_lo/hi')
                used |= {i, j}
                merged = True
                break
            # Pattern 2: _lo_SUFFIX / _hi_SUFFIX
            m_lo = re.match(r'^(.*?)_lo(_.*|/.*)$', name)
            m_hi = re.match(r'^(.*?)_hi(_.*|/.*)$', other)
            if (m_lo and m_hi
                    and m_lo.group(1) == m_hi.group(1)
                    and m_lo.group(2) == m_hi.group(2)):
                result.append(m_lo.group(1) + '_lo/hi' + m_lo.group(2))
                used |= {i, j}
                merged = True
                break
            # Pattern 3: xlo/ylo/zlo + xhi/yhi/zhi
            m_lo = re.match(r'^(.*?)xlo(/ylo)?(/zlo)?$', name)
            m_hi = re.match(r'^(.*?)xhi(/yhi)?(/zhi)?$', other)
            if m_lo and m_hi and m_lo.group(1) == m_hi.group(1):
                # txt_xlo = "xlo"
                # txt_ylo = "ylo" if m_lo.group(2) else ""
                # txt_zlo = "zlo" if m_lo.group(3) else ""
                # txt_xhi = "xhi"
                # txt_yhi = "yhi" if m_hi.group(2) else ""
                # txt_zhi = "zhi" if m_hi.group(3) else ""
                # txt = m_lo.group(1)
                # txt += '/'.join([txt_xlo, txt_ylo, txt_zlo, txt_xhi, txt_yhi, txt_zhi])
                txt = m_lo.group(1)
                txt += "xlo"
                txt += "/ylo" if m_lo.group(2) else ""
                txt += "/zlo" if m_lo.group(3) else ""
                txt += "/xhi"
                txt += "/yhi" if m_hi.group(2) else ""
                txt += "/zhi" if m_hi.group(3) else ""
                result.append(txt)
                used |= {i, j}
                merged = True
                break
            # Pattern: xmin,ymin,zmin & xmax,ymax,zmax
            sep_pat = r'([/,])'
            suffix_pat = r'(min|max)'
            pattern = fr'^(.*?)x{suffix_pat}({sep_pat}y{suffix_pat})?({sep_pat}z{suffix_pat})?(.*?)$'
            m1 = re.match(pattern, name)
            m2 = re.match(pattern, other)
            if m1 and m2 and m1.group(1) == m2.group(2):
                txt = m1.group(1)
                txt += 'xmin,ymin,zmin,xmax,ymax,zmax'
                result.append(txt)
                used |= {i, j}
                merged = True
                break
            # Pattern: PREFIX_x(...) & PREFIX_other

            # Pattern: <PREFIX>x<SUFFIX> & <PREFIX>y<SUFFIX> & <PREFIX>z<SUFFIX>
            pattern = r'^(.*?)([xyz])(.*?)'
            m1 = r'^'

        if not merged:
            result.append(name)

    return result

def merge_multiple_names(names: list[str]) -> list[str]:
    result = _merge_multiple_names(names)

    if len(result) != 1:
        print(f"\nmerge_multiple_names: failed to merge all names.\n  names  = {names}\n  result = {result}")
    if len(result) == 0:
        print(f"merge_multiple_names: WARNING: result = {result} is empty!")
    # if len(result) == 1 and len(names) != 1:
    #     print(f"\nmerge_multiple_names: iterative match successful\n  names  = {names}\n  result = {result}")

    # result = names

    if len(result) > 1:
        txt = ""
        if len(result) == 2:
            txt = result[0] + " and " + result[1]
        else:
            for i, elem in enumerate(result):
                if i == 0:
                    txt += elem
                # elif i == len(result) - 1:
                #     txt += " & " + elem
                else:
                    txt += ", " + elem
        result = [ txt ]
        print(f"  combined result = {result}")

    return result


# ── is_param_bullet ───────────────────────────────────────────────────────────

def is_param_bullet(name: str, rest: str) -> bool:
    """Return True if this bullet introduces a WarpX parameter definition."""
    # Skip value-assignment examples like ``my_constants.a0 = 3.0``
    if '=' in name:
        return False
    if '.' in name:
        return True
    if '<' in name and '>' in name:
        return True
    # Followed immediately by a parenthesised type annotation
    if re.match(r'\s*\(`', rest) or re.match(r'\s*\(``', rest):
        return True
    return False


# ── Span detection ────────────────────────────────────────────────────────────

def find_directive_spans(lines: list[str]) -> list[Span]:
    """Return one Span per top-level parameter bullet in the source.

    Each span is [start, end) where start is the bullet line and end is the
    first line that is no longer part of that bullet's body.
    """
    spans: list[Span] = []
    bullet_indent: int = -1
    body_indent: int = -1
    start: int = -1
    l_directive: bool = False

    for i, line in enumerate(lines):
        if l_directive:
            # Consume body: lines indented strictly deeper than the bullet,
            # including blank lines that precede more body content.
            if line.strip() == '':
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= bullet_indent or line_indent < body_indent:
                spans.append(Span(start, i))
                l_directive = False
                # print(f"\nfind_directive_spans:  span={start, i} bullet_indent={bullet_indent}")
        if not l_directive:
            m = PARAM_BULLET_RE.match(lines[i])
            if m and is_param_bullet(m.group(3), m.group(4)):
                bullet_indent = len(m.group(1))
                body_indent = detect_body_indent(lines, istart=i-1, bullet_indent=bullet_indent)
                if bullet_indent >= body_indent:
                    print(f"\nWARNING: bullet_indent >= b , {bullet_indent, body_indent}")
                start = i
                l_directive = True
    return spans


# ── Body stripping ────────────────────────────────────────────────────────────

def detect_body_indent(lines: list[str], *, istart=0, bullet_indent: int) -> int:
    """Return the column of the first non-blank body line.

    This is used to strip the source-level indentation when re-indenting
    body lines for the output directive.
    """
    for i in range(istart, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        col = len(line) - len(line.lstrip())
        if col > bullet_indent:
            return col
    return bullet_indent + 4  # fallback

def lstrip_lines(lines: list[str]) -> list[str]:
    return '\n'.join(lines).lstrip().split('\n')

def rstrip_lines(lines: list[str]) -> list[str]:
    return '\n'.join(lines).rstrip().split('\n')

def strip_lines(lines: list[str]) -> list[str]:
    return '\n'.join(lines).strip().split('\n')

# ── Main conversion ───────────────────────────────────────────────────────────

def convert(lines: list[str]) -> list[str]:
    """Replace parameter bullets with fv:var directives.

    Iterates through source lines once.  Lines belonging to a known directive
    span are replaced by the rendered directive (at the span's start line) or
    skipped (for the span's body lines).  All other lines pass through as-is.
    """
    spans = find_directive_spans(lines)

    # Map span-start index → Directive; collect all body-line indices to skip.
    span_starts: dict[int, Directive] = {}
    body_line_indices: set[int] = set()
    for span in spans:
        span_starts[span.start] = Directive(lines, span)
        body_line_indices.update(range(span.start + 1, span.end))

    out: list[str] = []
    for i, line in enumerate(lines):
        if i in body_line_indices:
            continue
        if i in span_starts:
            # if len(out) > 0 and out[-1].strip() != "":
            #     out.append("")
            out.extend(span_starts[i].render())
        else:
            out.append(line.rstrip('\n'))

    return out

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    with open('parameters_old.rst') as f:
        # lines = f.readlines()
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.pop()

    result = convert(lines)

    with open('source/usage/parameters.rst', 'w') as f:
        f.write('\n'.join(result) + '\n')

    n = sum(1 for ln in result if re.search(r'^\s*\.\. fv:var::', ln))
    print(f"Done: {len(lines)} source lines → {len(result)} output lines, {n} fv:var directives")


if __name__ == '__main__':
    main()
