# Heirachy of requirements:
# import pybtex
# import pybtex_docutils
# import sphinxcontrib.bibtex

import logging

import pybtex.style.template as template
import pybtex.richtext as richtext
from pybtex.richtext import Text
from pybtex.style.formatting.unsrt import Style as UnsrtStyle

# An brief introduction to custom BibTex formatting can be found in the Sphinx documentation:
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#bibtex-custom-formatting
#
# More details can be gleaned from looking at the pybtex dist-package files.
# Some examples include the following:
# BaseStyle class in pybtex/style/formatting/__init__.py
# UnsrtStyle class in pybtex/style/formating/unsrt.py
class WarpXBibStyle(UnsrtStyle):
    # This option makes the family name, i.e, "last" name, of an author to appear first.
    # default_name_style = 'lastfirst'

    def __init__(self, *args, **kwargs):
        # This option makes the given names of an author abbreviated to just initials.
        # Example: "Jean-Luc" becomes "J.-L."
        # Set 'abbreviate_names' to True before calling the superclass (BaseStyle class) initializer
        kwargs["abbreviate_names"] = True
        super().__init__(*args, **kwargs)

    def format_web_refs(self, e):
        # print("")
        # print("format_web_refs: start")
        try:
            result = super().format_web_refs(e)
            # print("  format_web_refs: success")
            # print(f"  format_web_refs: result = {result}")
            return result
        except Exception as exception:
            print("")
            print(f"  format_web_refs: print: Exception: {exception}")
            print("")
            logging.warning(f"  format_web_refs: logging.warrning: Exception: {exception}")
            raise exception
        # finally:
        #     print("format_web_refs: end")


    # def format_web_refs(self, e):
    #     url_node = template.optional[
    #         self.format_url(e),
    #         template.optional['(visited on ', template.field('urldate'), ')']
    #     ]
    #     eprint_node = template.optional[
    #         self.format_eprint(e),
    #     ]
    #     pubmed_node = template.optional[

    #     ]

    # Override
    def format_pubmed(self, e):
        # print("\nformat_pubmed: start")
        return format_href_fixed(
            prefix1='https://www.ncbi.nlm.nih.gov/pubmed/',
            prefix2='PMID:',
            field_name='pubmed',
        )

    # Override
    def format_doi(self, e):
        # print("\nformat_doi: start")
        try:
            return format_href_fixed(
                prefix1='https://doi.org/',
                prefix2='doi:',
                field_name='doi',
            )
        except Exception as exception:
            print("")
            print(f"  format_doi: print: Exception: {exception}")
            print("")
            logging.warning(f"  format_doi: logging.warrning: Exception: {exception}")
            return super().format_doi(e)

    def format_eprint(self, e):
        # based on urlbst format.eprint
        # print("\nformat_eprint: start")
        return format_href_fixed(
            prefix1='https://arxiv.org/abs/',
            prefix2='arXiv:',
            field_name='eprint',
        )

def format_href_fixed(
    prefix1: str,
    prefix2: str,
    field_name: str,
) -> template.Node:
    # node_raw = template.field(field_key, raw=True)
    # node1 = removeprefix_from_node(prefix=prefix1, node=node_raw)
    # node_fixed = removeprefix_from_node(prefix=prefix2, node=node1)

    def _fix_text(text: Text) -> Text:
        old_text = str(text)
        try:
            # print(f"\n_fix_text: start, text = {text}")
            text = removeprefix_from_Text(text, prefix=prefix1)
            text = removeprefix_from_Text(text, prefix=prefix2)
            text = removeprefix_from_Text(text, prefix=prefix1)
            text = removeprefix_from_Text(text, prefix=prefix2)
        except Exception as exception:
            print(f"  _fix_text: print: Exception {exception}")
            logging.warning(f"  _fix_text: logging.warning: Exception {exception}")
            text = Text(old_text)
            raise exception
        return text

    # node_fixed = template.field(field_name, apply_func=_fix_text, raw=True)
    # node1 = template.join[prefix1, node_fixed]
    # node2 = template.join[prefix2, node_fixed]

    def _remove_both_prefixes(str_: str):
        result = str_
        result = result.removeprefix(prefix1)
        result = result.removeprefix(prefix2)
        result = result.removeprefix(prefix1)
        result = result.removeprefix(prefix2)
        return result

    def _add_prefix(text: Text, prefix: str) -> Text:
        field_text_str = str(text)
        if field_text_str.startswith('https') and prefix.startswith('https'):
            # Defer to text
            result = text
        else:
            field_text_fixed_str = _remove_both_prefixes(field_text_str)
            result = Text(prefix + field_text_fixed_str)
        return result

    def _join_prefix_1(text: Text) -> Text:
        return _add_prefix(text, prefix1)

    def _join_prefix_2(text: Text) -> Text:
        return _add_prefix(text, prefix2)

    node1 = template.field(field_name, apply_func=_join_prefix_1, raw=True)
    node2 = template.field(field_name, apply_func=_join_prefix_2, raw=True)


    result = template.href[node1, node2]
    return result

# `@template.node`` decorator is defined as
# def node(f: Callable):
#     return Node(f.__name__, f)
# where `f`` is a function that outputs richtext.Text

# It is called in `Node.format_data`:
# def format_data(self, data):
#     return self.f(self.children, data, *self.args, **self.kwargs)
#
# Therefore, the function `f` must have the signature:
# f(children, data, *args, **kwargs) -> Text

def removeprefix_from_Text(txt: richtext.Text, prefix: str) -> richtext.Text:
    result = txt
    if prefix:
        try:
            prefix_str = str(prefix)
            txt_str = str(txt)
            txt_str_removed = txt_str.removeprefix(prefix_str)
            if txt_str_removed != txt_str:
                result = richtext.Text(txt_str_removed)
                print("")
                print("removeprefix_from_Text:")
                print(f"  text = {txt}")
                print(f"  prefix='{prefix}'")
                print(f"  result = '{result}'")
        except Exception as e:
            print("")
            print(f"removeprefix_from_Text('{txt}', prefix='{prefix}'):")
            print(f"  Exception: {e}")
            logging.warning(e)
    return result

@template.node
def _removeprefix_from_node(
    children,
    data,
    node: template.Node,
    prefix: str,
) -> richtext.Text:
    txt = node[children].format_data(data)
    result = removeprefix_from_Text(txt, prefix=prefix)
    return result

def removeprefix_from_node(
    node: template.Node,
    prefix: str
) -> template.Node:
    return _removeprefix_from_node(node=node, prefix=prefix)

@template.node
def join_avoid_redundant_prefix(
    children,
    data,
    *args,
    **kwargs
) -> richtext.Text:

    join_node: template.Node
    result: richtext.Text
    extra_text: richtext.Text

    join_node = template.join(*args, **kwargs)
    result = join_node[children].format_data(data)

    try:
        child0 = children[0]
        if child0:
            prefix = str(child0)
            extra_text = join_node[children[1:]].format_data(data)
            extra_text = removeprefix_from_Text(extra_text, prefix=prefix)
            new_text = richtext.Text(prefix + str(extra_text))
        if str(new_text) != str(result):
            old_result_str = str(result)
            result = new_text
            print(f"\njoin_remove_redundant_prefix:\n  removing extra prefix='{prefix}' in '{old_result_str}'\n  result = '{result}'")
    except Exception as e:
        print(f"\njoin_remove_redundant_prefix:\n  Exception: {e}")
        logging.warning(e)

    return result

# def format_href_node(
#     children,
#     data,
#     prefix1: str,
#     prefix2: str,
#     field_name: str,
# ) -> Text:

#     # if field_text.startswith('https'):
#     #     if prefix1.startswith('https'):
#     #         node1 =

#     @template.node
#     def node1(children, data) -> Text:
#         assert not children
#         field_text: Text = template.field(field_name, raw=False).format_data(data)
#         field_text_str: str = str(field_text)
#         result = field_text

#         if field_text_str.startswith('https') and prefix1.startswith('https'):
#             return field_text
#         else:
#             result_str = field_text_str.removeprefix(prefix1)
#             result_str = field_text_str.removeprefix(prefix2)
#             result_str = field_text_str.removeprefix(prefix1)
#             result_str = field_text_str.removeprefix(prefix2)
#             result = Text(prefix1 + result_str)
#         return result

#     result = template.href[node1, node2]
#     return result
