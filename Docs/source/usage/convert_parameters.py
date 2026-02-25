"""
Convert WarpX parameters.rst to use fv:var directives.

Only top-level param bullets are converted; nested param bullets inside
description bodies are left as raw RST (no recursive conversion).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple


# ── Regex ─────────────────────────────────────────────────────────────────────

# Matches a RST bullet that introduces a parameter:
#   group 1 – leading whitespace (indent depth)
#   group 2 – bullet character (* or -)
#   group 3 – parameter name (inside double backticks)
#   group 4 – remainder of line after the closing ``
PARAM_BULLET_RE = re.compile(r'^( *)([*\-]) ``([^`]+)``(.*)', re.DOTALL)


# ── Span ──────────────────────────────────────────────────────────────────────

class Span(NamedTuple):
    """Half-open line range [start, end) within the source file."""
    start: int
    end: int


# ── Directive dataclass ───────────────────────────────────────────────────────

@dataclass
class Directive:
    """All information extracted from one parameter bullet.

    A single source bullet may produce multiple directives when its names
    don't all merge (e.g. save_particles_at_xlo/hi + save_particles_at_eb).
    Each name in `names` will be rendered as a separate ``.. fv:var::`` block
    sharing the same type, default, and body.
    """

    # Final names after lo/hi merging; each becomes its own fv:var directive
    names: list[str]

    # RST :type: value (may be empty)
    type_str: str

    # RST :default: value (may be empty)
    default_str: str

    # Body lines as they appear in the source (original indentation, no newlines)
    raw_body: list[str]

    # Body lines re-indented for output (source indent stripped, ready for
    # ``indent + "    " + line`` in render_directive)
    body: list[str]

    # Source line range [start, end) that this directive was built from
    source_span: Span

    # Number of leading spaces on the originating bullet line
    bullet_indent: int


# ── Type normalisation ────────────────────────────────────────────────────────

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
    if not s:
        return s
    s = s.strip()
    # ``0`` or ``1`` → bool
    s = re.sub(r'[`]*0[`]*\s+or\s+[`]*1[`]*', 'bool', s)
    # Normalise standalone type words
    for old, new in WORD_NORMS.items():
        s = re.sub(rf'(?<!\w){re.escape(old)}(?!\w)', new, s, flags=re.IGNORECASE)
    # Fix double-backtick artifacts: ``bool`` → `bool`
    s = re.sub(r'``(int|float|str|bool)``', r'`\1`', s)
    return s


# ── Parenthesis finder ────────────────────────────────────────────────────────

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


# ── Meta parsing ──────────────────────────────────────────────────────────────

def parse_meta(s: str) -> tuple[str, str]:
    """Parse the interior of a parenthesised annotation into (type_str, default_str).

    Recognises patterns such as:
        `float`; in seconds
        `float`; default: 1.0
        `int`, 0 by default
        ``0`` or ``1``; default is ``1`` for true
    """
    s = s.strip()
    # "; default …" or ", default …"
    m = re.search(r'(?:;|,)\s*default(?:\s+is)?:?\s*(.*)', s, re.IGNORECASE)
    if m:
        return normalise_type(s[:m.start()].strip()), m.group(1).strip().rstrip(')')
    # "… X by default"
    m2 = re.search(
        r'[,;]\s*(`[^`]+`|``[^`]+``|\S+)\s+by\s+default\s*$', s, re.IGNORECASE
    )
    if m2:
        return normalise_type(s[:m2.start()].strip()), m2.group(1).strip()
    return normalise_type(s), ''


# ── Bullet annotation ─────────────────────────────────────────────────────────

class BulletAnnotation(NamedTuple):
    """Parsed result of everything after the first ``name`` on a bullet line."""
    extra_names: list[str]  # co-listed names, e.g. from "and ``foo.hi``"
    type_str: str
    default_str: str
    inline_desc: str        # descriptive text on the bullet line itself


def parse_bullet_annotation(rest: str) -> BulletAnnotation:
    """Parse everything after the opening ``name`` on a parameter bullet line.

    Handles the following source patterns (non-exhaustive):
        , ``co.name`` and ``other.name`` (`type`; default: X) desc…
        (`type`; default: X) optional
        optional (default: X) desc…
        (`string`: e.g. ``...``)
        (``0`` or ``1``; default is ``1`` for true)
        plain description with no annotation
    """
    s = rest.strip()

    # ── Collect additional co-listed names ────────────────────────────────────
    extra_names: list[str] = []
    while True:
        m = re.match(r'^(?:,\s*|and\s+|,\s*and\s+)``([^`]+)``\s*(.*)', s, re.DOTALL)
        if m:
            extra_names.append(m.group(1).strip())
            s = m.group(2).strip()
        else:
            break

    type_str = default_str = inline_desc = ''

    if s.startswith('('):
        # Main parenthesised annotation
        end = find_paren_end(s)
        type_str, default_str = parse_meta(s[1:end])
        after = s[end + 1:].strip()
        # Strip a bare "optional" that sometimes follows
        after = re.sub(r'^optional\b', '', after, flags=re.IGNORECASE).strip()
        # A second parenthesis may carry the default when the first didn't
        if after.startswith('(') and not default_str:
            end2 = find_paren_end(after)
            dm = re.match(r'default:?\s*(.*)', after[1:end2].strip(), re.IGNORECASE)
            if dm:
                default_str = dm.group(1).strip()
            inline_desc = after[end2 + 1:].strip()
        else:
            inline_desc = after

    elif s.lower().startswith('optional'):
        remainder = s[len('optional'):].strip()
        dm = re.match(r'\(default:?\s*(.*?)\)(.*)', remainder, re.IGNORECASE | re.DOTALL)
        if dm:
            default_str = dm.group(1).strip()
            inline_desc = dm.group(2).strip()
        else:
            inline_desc = remainder

    elif s:
        inline_desc = s

    # Strip a lone trailing "." left over from e.g. "(`type`, default: X)."
    inline_desc = inline_desc.strip().lstrip('.').strip()

    return BulletAnnotation(
        extra_names=extra_names,
        type_str=type_str,
        default_str=default_str,
        inline_desc=inline_desc,
    )


# ── lo/hi name merging ────────────────────────────────────────────────────────

def merge_lo_hi_names(names: list[str]) -> list[str]:
    """Collapse lo/hi name pairs into a single combined name.

    Three patterns are handled:
      1. Simple pair:       foo_lo + foo_hi              → foo_lo/hi
      2. Suffixed pair:     foo_lo_x/y/z + foo_hi_x/y/z → foo_lo/hi_x/y/z
      3. Multi-axis pair:   at_xlo/ylo/zlo + at_xhi/yhi/zhi
                             → at_xlo/hi_ylo/hi_zlo/hi

    Names that do not participate in any merge are returned unchanged.
    When three names are present and two merge, the merged name and the
    unmerged name are both returned (e.g. [merged, at_eb]).
    """
    if len(names) < 2:
        return names

    result: list[str] = []
    used: set[int] = set()

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
                axes = (
                    ('_ylo/hi' if m_lo.group(2) else '')
                    + ('_zlo/hi' if m_lo.group(3) else '')
                )
                result.append(m_lo.group(1) + 'xlo/hi' + axes)
                used |= {i, j}
                merged = True
                break
        if not merged:
            result.append(name)

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
    i = 0
    while i < len(lines):
        m = PARAM_BULLET_RE.match(lines[i])
        if m and is_param_bullet(m.group(3), m.group(4)):
            bullet_indent = len(m.group(1))
            start = i
            i += 1
            # Consume body: lines indented strictly deeper than the bullet,
            # including blank lines that precede more body content.
            while i < len(lines):
                bl = lines[i]
                if bl.strip() == '':
                    # Peek ahead past any blanks
                    la = i + 1
                    while la < len(lines) and lines[la].strip() == '':
                        la += 1
                    if la < len(lines):
                        next_indent = len(lines[la]) - len(lines[la].lstrip())
                        if next_indent > bullet_indent:
                            i += 1
                            continue
                    break
                if len(bl) - len(bl.lstrip()) <= bullet_indent:
                    break
                i += 1
            spans.append(Span(start, i))
        else:
            i += 1
    return spans


# ── Body stripping ────────────────────────────────────────────────────────────

def detect_body_indent(lines: list[str], body_start: int, bullet_indent: int) -> int:
    """Return the column of the first non-blank body line.

    This is used to strip the source-level indentation when re-indenting
    body lines for the output directive.
    """
    j = body_start
    while j < len(lines) and lines[j].strip() == '':
        j += 1
    if j < len(lines):
        col = len(lines[j]) - len(lines[j].lstrip())
        if col > bullet_indent:
            return col
    return bullet_indent + 4  # fallback


# ── Directive construction ────────────────────────────────────────────────────

def build_directive(lines: list[str], span: Span) -> Directive:
    """Construct a Directive from the source lines covered by span."""
    bullet_line = lines[span.start]
    m = PARAM_BULLET_RE.match(bullet_line)
    assert m is not None  # guaranteed by find_directive_spans

    bullet_indent = len(m.group(1))
    first_name = m.group(3).strip()
    rest = m.group(4).rstrip('\n')

    annotation = parse_bullet_annotation(rest)
    names = merge_lo_hi_names([first_name] + annotation.extra_names)

    # Raw body: every source line after the bullet, newlines stripped
    raw_body: list[str] = [lines[k].rstrip('\n') for k in range(span.start + 1, span.end)]

    # Re-indented body: strip source indent, leaving content ready to be
    # prefixed with ``bullet_indent * ' ' + '    '`` in render_directive.
    strip = detect_body_indent(lines, span.start + 1, bullet_indent)
    body: list[str] = []
    if annotation.inline_desc:
        body.append(annotation.inline_desc)
    for raw in raw_body:
        if raw.strip() == '':
            body.append('')
        else:
            col = len(raw) - len(raw.lstrip())
            body.append(raw[min(strip, col):])
    # Drop trailing blank lines
    remove_trailing_blank_lines(body)

    return Directive(
        names=names,
        type_str=annotation.type_str,
        default_str=annotation.default_str,
        raw_body=raw_body,
        body=body,
        source_span=span,
        bullet_indent=bullet_indent,
    )

def remove_leading_blank_lines(lines: list[str]):
    while lines and lines[-1].strip() == '':
        lines.pop(-1)
    return lines

def remove_trailing_blank_lines(lines: list[str]):
    while lines and lines[0].strip() == '':
        lines.pop(0)
    return lines

def strip_blank_lines(lines: list[str]):
    remove_leading_blank_lines(lines)
    remove_trailing_blank_lines(lines)
    return lines

# ── RST output rendering ──────────────────────────────────────────────────────

def render_directive(d: Directive) -> list[str]:
    """Render a Directive to output lines.

    One ``.. fv:var::`` block is emitted per name in d.names, all sharing
    the same type, default, and body.  Blocks are separated by blank lines.
    """
    indent = ' ' * d.bullet_indent
    out: list[str] = []

    l_past_first_name = False
    for name in d.names:
        if l_past_first_name:
            out.append('')
        l_past_first_name = True
        out.append(f'{indent}.. fv:var:: {name}')
        if d.type_str:
            out.append(f'{indent}    :type: {d.type_str}')
        if d.default_str:
            out.append(f'{indent}    :default: {d.default_str}')
        if '\n'.join(d.body).strip():
            out.append('')
            for line in d.body:
                out.append(f'{indent}    {line}' if line.strip() else '')

    while out and out[0].strip == '':
        out.pop(0)
    while out and out[-1].strip() == '':
        out.pop()

    return out


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
        span_starts[span.start] = build_directive(lines, span)
        body_line_indices.update(range(span.start + 1, span.end))

    out: list[str] = []
    for i, line in enumerate(lines):
        if i in body_line_indices:
            continue
        if i in span_starts:
            if len(out) > 0 and out[-1].strip() != "":
                out.append("")
            out.extend(render_directive(span_starts[i]))
        else:
            out.append(line.rstrip('\n'))

    return out

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    with open('parameters_old.rst') as f:
        lines = f.readlines()

    result = convert(lines)

    with open('parameters.rst', 'w') as f:
        f.write('\n'.join(result) + '\n')

    n = sum(1 for ln in result if re.search(r'^\s*\.\. fv:var::', ln))
    print(f"Done: {len(lines)} source lines → {len(result)} output lines, {n} fv:var directives")


if __name__ == '__main__':
    main()
