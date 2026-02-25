"""
Convert WarpX parameters.rst to use fv:var directives.
"""
import re

PARAM_BULLET_RE = re.compile(r'^( *)([*\-]) ``([^`]+)``(.*)', re.DOTALL)


def is_param_bullet(name, rest):
    if '=' in name:
        return False
    if '.' in name:
        return True
    if '<' in name and '>' in name:
        return True
    if re.match(r'\s*\(`', rest) or re.match(r'\s*\(``', rest):
        return True
    return False


def find_paren_end(s):
    depth = 0
    ci = 0
    while ci < len(s):
        if s[ci:ci+2] == '``':
            ci += 2
            end_bt = s.find('``', ci)
            ci = end_bt + 2 if end_bt != -1 else len(s)
            continue
        if s[ci] == '(':
            depth += 1
        elif s[ci] == ')':
            depth -= 1
            if depth == 0:
                return ci
        ci += 1
    return len(s) - 1


# ── Type normalisation ────────────────────────────────────────────────────────

WORD_NORMS = {
    'integer': 'int', 'integers': 'int',
    'boolean': 'bool',
    'double': 'float', 'doubles': 'float',
    'string': 'str', 'strings': 'str',
}


def norm_word(w):
    return WORD_NORMS.get(w.lower().strip('`'), w.strip('`'))


def normalise_type(s):
    if not s:
        return s
    s = s.strip()
    s = re.sub(r'[`]*0[`]*\s+or\s+[`]*1[`]*', 'bool', s)

    def list_sub(m):
        inner_raw = m.group(1).strip().strip('`')
        inner = norm_word(inner_raw)
        return f'list[{inner}]'

    s = re.sub(r'\bstrings\b', 'list[str]', s)

    for old, new in [('integers', 'int'), ('integer', 'int'),
                     ('boolean', 'bool'),
                     ('doubles', 'float'), ('double', 'float'),
                     ('string', 'str')]:
        s = re.sub(rf'(?<!\w){old}(?!\w)', new, s, flags=re.IGNORECASE)

    # Fix double-backtick wrapping artifacts: ``bool`` → `bool` etc.
    s = re.sub(r'``(int|float|str|bool)``', r'`\1`', s)
    # Remove inner backticks in list params: list[`str`] → list[str]
    s = re.sub(r'list\[`(\w+)`\]', r'list[\1]', s)
    # Wrap list[X] in backticks if not already
    s = re.sub(r'(?<!`)list\[(\w+)\](?!`)', r'`list[\1]`', s)
    return s


# ── Meta parsing ──────────────────────────────────────────────────────────────

def parse_meta(s):
    s = s.strip()
    m = re.search(r'(?:;|,)\s*default(?:\s+is)?:?\s*(.*)', s, re.IGNORECASE)
    if m:
        return normalise_type(s[:m.start()].strip()), m.group(1).strip().rstrip(')')
    m2 = re.search(r'[,;]\s*(`[^`]+`|``[^`]+``|\S+)\s+by\s+default\s*$', s, re.IGNORECASE)
    if m2:
        return normalise_type(s[:m2.start()].strip()), m2.group(1).strip()
    return normalise_type(s), ''


def parse_names_from_rest(rest):
    s = rest.strip()
    extra_names = []
    while True:
        m = re.match(r'^(?:,\s*|and\s+|,\s*and\s+)``([^`]+)``\s*(.*)', s, re.DOTALL)
        if m:
            extra_names.append(m.group(1).strip())
            s = m.group(2).strip()
        else:
            break

    type_str = default_str = extra_desc = ''

    if s.startswith('('):
        end = find_paren_end(s)
        type_str, default_str = parse_meta(s[1:end])
        after = s[end+1:].strip()
        after = re.sub(r'^optional\s*', '', after, flags=re.IGNORECASE).strip()
        if after.startswith('(') and not default_str:
            end2 = find_paren_end(after)
            dm = re.match(r'default:?\s*(.*)', after[1:end2].strip(), re.IGNORECASE)
            if dm:
                default_str = dm.group(1).strip()
            extra_desc = after[end2+1:].strip()
        elif after:
            extra_desc = after
    elif s.lower().startswith('optional'):
        remainder = s[len('optional'):].strip()
        dm = re.match(r'\(default:?\s*(.*?)\)(.*)', remainder, re.IGNORECASE | re.DOTALL)
        if dm:
            default_str = dm.group(1).strip()
            extra_desc = dm.group(2).strip()
        elif remainder:
            extra_desc = remainder
    elif s:
        extra_desc = s

    # Strip trailing punctuation-only extra_desc (e.g. a lone ".")
    extra_desc = extra_desc.strip().lstrip('.').strip()

    return extra_names, type_str, default_str, extra_desc


# ── Directive builder ─────────────────────────────────────────────────────────

def make_directive(name, type_str, default_str, body_lines, indent='', body_extra='    '):
    """body_extra: whitespace prepended to each body line inside the directive."""
    out = [f'{indent}.. fv:var:: {name}']
    if type_str:
        out.append(f'{indent}    :type: {type_str}')
    if default_str:
        out.append(f'{indent}    :default: {default_str}')
    if body_lines:
        out.append('')
        for line in body_lines:
            out.append(f'{indent}{body_extra}{line}' if line.strip() else '')
        while out and out[-1].strip() == '':
            out.pop()

    # out.append('')

    # out = ('\n'.join(out).rstrip()).split('\n')

    # if "max_step" in out[0] or "stop_time" in out[0]:
    #     print("")
    #     print(f"out[0] = {out[0]}, out[-1] = {out[-1]}")

    return out


# ── Body collection ───────────────────────────────────────────────────────────

def detect_strip_amount(lines, i, bullet_indent):
    """Scan ahead to find the actual indentation of the first body line."""
    j = i
    while j < len(lines) and lines[j].strip() == '':
        j += 1
    if j < len(lines):
        body_indent = len(lines[j]) - len(lines[j].lstrip())
        if body_indent > bullet_indent:
            return body_indent
    # Fallback: bullet_indent + 4 for *, +2 for -
    return bullet_indent + 4


def collect_body(lines, i, bullet_indent, strip_amount):
    """Collect ALL lines deeper than bullet_indent (including nested param bullets).
    strip_amount: how many leading chars to strip from each line."""
    body = []
    while i < len(lines):
        bl = lines[i]
        if bl.strip() == '':
            la = i + 1
            while la < len(lines) and lines[la].strip() == '':
                la += 1
            if la < len(lines):
                ni = len(lines[la]) - len(lines[la].lstrip())
                if ni > bullet_indent:
                    body.append('')
                    i += 1
                    continue
            break
        line_indent = len(bl) - len(bl.lstrip())
        if line_indent <= bullet_indent:
            break
        # Strip at most strip_amount chars (some continuation lines may be less indented)
        body.append(bl[min(strip_amount, line_indent):].rstrip('\n'))
        i += 1
    while body and body[-1].strip() == '':
        body.pop()
    return body, i


# ── Main (recursive) ──────────────────────────────────────────────────────────

# ── lo/hi name merging ────────────────────────────────────────────────────────

def merge_lo_hi_names(names):
    """Merge a list of parameter names into combined lo/hi form where applicable.

    Rules (applied in order):
    - Two names with the same prefix, one ending _lo and one _hi:
        e.g. ['foo.bar_lo', 'foo.bar_hi'] → ['foo.bar_lo/hi']
    - Two names where one ends in _lo suffix pattern and the other _hi,
        with further suffixes like _x/y/z:
        e.g. ['foo.pot_lo_x/y/z', 'foo.pot_hi_x/y/z'] → ['foo.pot_lo/hi_x/y/z']
    - Complex multi-axis cases like xlo/ylo/zlo + xhi/yhi/zhi:
        e.g. ['sp.at_xlo/ylo/zlo', 'sp.at_xhi/yhi/zhi', 'sp.at_eb']
         → ['sp.at_xlo/hi_ylo/hi_zlo/hi', 'sp.at_eb']
    - Names that don't match any pattern are left unchanged.
    """
    if len(names) < 2:
        return names

    result = []
    used = set()

    for i, name in enumerate(names):
        if i in used:
            continue

        merged = False
        for j, other in enumerate(names):
            if j <= i or j in used:
                continue

            # Case 1: simple _lo / _hi pair with same prefix
            # e.g. foo._lo and foo._hi
            import re as _re
            m_lo = _re.match(r'^(.*?)_lo$', name)
            m_hi = _re.match(r'^(.*?)_hi$', other)
            if m_lo and m_hi and m_lo.group(1) == m_hi.group(1):
                result.append(m_lo.group(1) + '_lo/hi')
                used.add(i); used.add(j)
                merged = True
                break

            # Case 2: _lo_SUFFIX / _hi_SUFFIX (e.g. potential_lo_x/y/z)
            m_lo = _re.match(r'^(.*?)_lo(_.*|/.*)?$', name)
            m_hi = _re.match(r'^(.*?)_hi(_.*|/.*)?$', other)
            if (m_lo and m_hi
                    and m_lo.group(1) == m_hi.group(1)
                    and m_lo.group(2) == m_hi.group(2)
                    and m_lo.group(2)):  # must have a suffix
                suffix = m_lo.group(2)
                result.append(m_lo.group(1) + '_lo/hi' + suffix)
                used.add(i); used.add(j)
                merged = True
                break

            # Case 3: xlo/ylo/zlo + xhi/yhi/zhi multi-axis
            # e.g. sp.save_particles_at_xlo/ylo/zlo + sp.save_particles_at_xhi/yhi/zhi
            m_lo = _re.match(r'^(.*?)xlo(/ylo)?(/zlo)?$', name)
            m_hi = _re.match(r'^(.*?)xhi(/yhi)?(/zhi)?$', other)
            if m_lo and m_hi and m_lo.group(1) == m_hi.group(1):
                prefix = m_lo.group(1)
                axes = ''
                if m_lo.group(2): axes += '_ylo/hi'
                if m_lo.group(3): axes += '_zlo/hi'
                result.append(prefix + 'xlo/hi' + axes)
                used.add(i); used.add(j)
                merged = True
                break

        if not merged:
            result.append(name)

    return result


def convert(lines: list[str]) -> list[str]:
    """Convert lines to fv:var directives, recursively for nested params."""
    out: list[str] = []
    i = 0

    lprintcount = 3

    while i < len(lines):
        line = lines[i]
        m = PARAM_BULLET_RE.match(line)
        if m and is_param_bullet(m.group(3), m.group(4)):
            debugdict = {}
            debugdict['istart'] = i
            debugdict['m'] = m

            bullet_indent = len(m.group(1))
            bullet_char = m.group(2)
            first_name = m.group(3).strip()
            rest = m.group(4).rstrip('\n')
            extra_names, type_str, default_str, extra_desc = parse_names_from_rest(rest)
            names = merge_lo_hi_names([first_name] + extra_names)
            i += 1
            # Detect actual body indentation from first body line
            strip_amount = detect_strip_amount(lines, i, bullet_indent)
            # body_extra for make_directive:
            # If body starts at bullet_indent+4 → 4 spaces; at bullet_indent+2 → 6 spaces
            body_offset = strip_amount - bullet_indent
            body_extra = '    ' if body_offset >= 4 else '      '
            raw_body, i = collect_body(lines, i, bullet_indent, strip_amount)

            debugdict['icollect_body'] = i

            converted_body = convert(raw_body)
            if extra_desc:
                converted_body.insert(0, extra_desc)
            dir_indent = ' ' * bullet_indent
            for name in names:
                out.extend(make_directive(name, type_str, default_str, converted_body,
                                          dir_indent, body_extra))
                out.append('')

            debugdict['out[0]'] = out[0]
            debugdict['out[-2:]'] = out[-2:]

            while i < len(line) and line[i].strip() == '':
                i += 1

            debugdict['ifinal'] = i

            if lprintcount > 0:
                lprintcount -= 1
                print("")
                print("debugdict:")
                for k, v in debugdict.items():
                    print(f"{k}: {v}")
        else:
            out.append(line.rstrip('\n'))
            i += 1
    return out


with open('parameters_old.rst') as f:
    lines = f.readlines()

result = convert(lines)

# Collapse multiple consecutive blank lines to one
collapsed = []
prev_blank = False
for line in result:
    is_blank = line.strip() == ''
    if is_blank and prev_blank:
        continue
    collapsed.append(line)
    prev_blank = is_blank

with open('parameters.rst', 'w') as f:
    f.write('\n'.join(collapsed) + '\n')

n = sum(1 for l in collapsed if re.search(r'^\s*\.\. fv:var::', l))
print(f"Done: {len(lines)} -> {len(collapsed)} lines, {n} fv:var directives")
