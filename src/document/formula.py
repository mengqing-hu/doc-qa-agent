"""Extract Word equations and recognize formula-like PDF regions."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from docx.text.paragraph import Paragraph
GREEK_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "θ": r"\theta",
    "η": r"\eta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "φ": r"\phi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Λ": r"\Lambda",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ω": r"\Omega",
}
SYMBOL_LATEX = {
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\ne",
    "≈": r"\approx",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "√": r"\sqrt",
    "∞": r"\infty",
    "∇": r"\nabla",
    "∂": r"\partial",
    "×": r"\times",
    "÷": r"\div",
    "⋅": r"\cdot",
    "−": "-",
}


def extract_paragraph_text(paragraph: Paragraph) -> str:
    """Return Word paragraph text with OMML equations represented as LaTeX."""
    fragments = [
        _word_element_to_text(child)
        for child in paragraph._p
        if _local_name(child) != "pPr"
    ]
    return _normalize_text("".join(fragments))


def extract_table_cell_text(cell: Any) -> str:
    """Return a table cell's paragraphs while preserving embedded OMML equations."""
    return "\n".join(
        text for paragraph in cell.paragraphs if (text := extract_paragraph_text(paragraph))
    )


def omml_to_latex(element: Any) -> str:
    """Convert common Office Math Markup Language structures to LaTeX."""
    name = _local_name(element)
    if name in {"oMath", "e", "num", "den", "sub", "sup", "deg", "lim", "fName"}:
        return _omml_children_text(element)
    if name == "r":
        return "".join(
            child.text or ""
            for child in element
            if _local_name(child) == "t"
        )
    if name == "f":
        return rf"\frac{{{_omml_child_text(element, 'num')}}}{{{_omml_child_text(element, 'den')}}}"
    if name == "sSub":
        return rf"{_omml_child_text(element, 'e')}_{{{_omml_child_text(element, 'sub')}}}"
    if name == "sSup":
        return rf"{_omml_child_text(element, 'e')}^{{{_omml_child_text(element, 'sup')}}}"
    if name == "sSubSup":
        return (
            rf"{_omml_child_text(element, 'e')}_{{{_omml_child_text(element, 'sub')}}}"
            rf"^{{{_omml_child_text(element, 'sup')}}}"
        )
    if name == "rad":
        degree = _omml_child_text(element, "deg")
        expression = _omml_child_text(element, "e")
        return rf"\sqrt[{degree}]{{{expression}}}" if degree else rf"\sqrt{{{expression}}}"
    if name == "nary":
        properties = _omml_child(element, "naryPr")
        character = _omml_property_value(properties, "chr") or "∑"
        operator = _latex_symbols(character)
        lower = _omml_child_text(element, "sub")
        upper = _omml_child_text(element, "sup")
        limits = (rf"_{{{lower}}}" if lower else "") + (rf"^{{{upper}}}" if upper else "")
        return f"{operator}{limits} {_omml_child_text(element, 'e')}"
    if name in {"limLow", "limUpp"}:
        base = _omml_child_text(element, "e")
        limit = _omml_child_text(element, "lim")
        marker = "_" if name == "limLow" else "^"
        return rf"{base}{marker}{{{limit}}}"
    if name == "func":
        return rf"{_omml_child_text(element, 'fName')}({_omml_child_text(element, 'e')})"
    if name == "d":
        properties = _omml_child(element, "dPr")
        begin = _omml_property_value(properties, "begChr") or "("
        end = _omml_property_value(properties, "endChr") or ")"
        return f"{begin}{_omml_child_text(element, 'e')}{end}"
    if name == "m":
        rows = [
            " & ".join(_omml_children_text(cell) for cell in row if _local_name(cell) == "e")
            for row in element
            if _local_name(row) == "mr"
        ]
        return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"
    if name in {"bar", "acc", "groupChr"}:
        return _omml_child_text(element, "e")
    return _omml_children_text(element)


def _word_element_to_text(element: Any) -> str:
    """Render text-bearing Word XML nodes without treating formatting as content."""
    name = _local_name(element)
    if name == "oMath":
        latex = _normalize_latex(omml_to_latex(element))
        return f"${latex}$" if latex else ""
    if name == "oMathPara":
        latex = _normalize_latex(_omml_children_text(element))
        return f"\n$${latex}$$\n" if latex else ""
    if name in {"t", "delText"}:
        return element.text or ""
    if name == "tab":
        return "\t"
    if name in {"br", "cr"}:
        return "\n"
    return "".join(_word_element_to_text(child) for child in element)


def _omml_children_text(element: Any) -> str:
    """Return the concatenated content of non-property OMML child elements."""
    return "".join(
        omml_to_latex(child)
        for child in element
        if not _local_name(child).endswith("Pr")
    )


def _omml_child(element: Any, name: str) -> Any | None:
    """Find the direct OMML child with the requested local name."""
    return next((child for child in element if _local_name(child) == name), None)


def _omml_child_text(element: Any, name: str) -> str:
    """Return the converted content of a direct OMML child."""
    child = _omml_child(element, name)
    return _normalize_latex(omml_to_latex(child)) if child is not None else ""


def _omml_attribute(element: Any | None, name: str) -> str | None:
    """Read an OMML attribute regardless of its XML namespace."""
    if element is None:
        return None
    return next(
        (value for key, value in element.attrib.items() if _local_name_from_tag(key) == name),
        None,
    )


def _omml_property_value(element: Any | None, name: str) -> str | None:
    """Read an OMML property value stored in a named child element."""
    child = _omml_child(element, name) if element is not None else None
    return _omml_attribute(child, "val") if child is not None else None


def _normalize_latex(text: str) -> str:
    """Normalize whitespace and common Office Math symbols in LaTeX output."""
    converted = "".join(
        character for character in _latex_symbols(text) if unicodedata.category(character) != "Cf"
    )
    return re.sub(r"\s+", " ", converted).strip()


def _latex_symbols(text: str) -> str:
    """Replace Unicode mathematical symbols with their LaTeX equivalents."""
    converted: list[str] = []
    for index, character in enumerate(text):
        replacement = SYMBOL_LATEX.get(character, GREEK_LATEX.get(character, character))
        converted.append(replacement)
        if (
            replacement != character
            and replacement.startswith("\\")
            and index + 1 < len(text)
            and text[index + 1].isalpha()
        ):
            converted.append(" ")
    return "".join(converted)


def _normalize_text(text: str) -> str:
    """Normalize spaces while retaining explicit paragraph line breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _local_name(element: Any) -> str:
    """Return an XML element's namespace-independent local name."""
    return _local_name_from_tag(str(element.tag))


def _local_name_from_tag(tag: str) -> str:
    """Return the local name portion of a Clark-notation XML tag."""
    return tag.rsplit("}", maxsplit=1)[-1]
