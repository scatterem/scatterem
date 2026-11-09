"""Sphinx configuration of the scatterem documentation site.

Build from the repository root (CI does the same)::

    pip install -r docs/requirements.txt
    sphinx-build -W --keep-going -n -c docs -d _build/doctrees . _build/html

The source directory is the repository root, so the pages keep the paths of
the published site: ``index.html`` and ``docs/<page>.html``. Only
``index.md`` and ``docs/*.md`` are pages.

The build does not execute code blocks. The API reference (``docs/api.md``)
is autodoc of the installed package. The build fails when a public module
exports a name that the reference does not show, and when the reference shows
a name of an internal module.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Document the checkout, not some other installed copy of the package.
sys.path.insert(0, str(ROOT))

import scatterem  # noqa: E402

# -- Project ------------------------------------------------------------------

project = "scatterem"
author = "Shengbo You, Nikita Palatkin, Sihan Shao and Philipp Pelz"
copyright = "2026, the scatterem authors"  # noqa: A001
release = scatterem.__version__
version = release

# -- General --------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",  # writes .nojekyll into the output
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
include_patterns = ["index.md", "docs/*.md"]
exclude_patterns = ["_build"]
templates_path: list[str] = []

# Every cross-reference that does not resolve is a warning, and the build runs
# with -W, so a renamed object fails the build instead of leaving a dead link.
nitpicky = True

#: Targets that cannot resolve, each with its reason. Keep the list short: it
#: is the only way around ``nitpicky``.
nitpick_ignore_regex = [
    # Napoleon turns each word of a numpydoc type field into a class reference.
    # Type fields are prose, so these cannot resolve: words and phrases
    # ("array_like", "default True", "str to float", "the 12 slots"), single
    # lower-case words ("sequence", "mapping", "float32"), shape symbols
    # ("Ny", "B_S", "H"), numbers, literal choice sets and the split halves of
    # subscripted types ("tuple[str"). Explicit roles (:func:, :meth:, ...)
    # in the text are not covered here and stay strict.
    (r"py:class", r".*\s.*"),
    (r"py:class", r"[a-z_][a-z0-9_-]*"),
    (r"py:class", r"[A-Z][A-Za-z]?(?:_[A-Za-z]+)?|N[A-Za-z]*"),
    (r"py:class", r"[0-9.e-]+"),
    (r"py:class", r"default=.*"),
    (r"py:class", r".*[\"'{}\[\]].*"),
    # torch, numpy and matplotlib annotate with names that their inventories
    # do not hold under that spelling, or hold as another object type.
    (r"py:class", r"(?:torch\.)?(?:Tensor|Size|Generator)"),
    (r"py:class", r"(?:torch\.nn\.)?(?:Module|Parameter)"),
    (r"py:class", r"(?:np\.|numpy\.)?(?:ndarray|NDArray|typing\.NDArray)"),
    (r"py:class", r"numpy\.(?:u?int|float|complex|bool)\d*"),
    (r"py:class", r"numpy\._typing.*"),
    (r"py:class", r"ArrayLike"),
    (r"py:class", r"Path"),
    (r"py:class", r"Colormap|RcKeyType"),
    # NVIDIA Warp publishes no Sphinx inventory; the Warp kernels annotate
    # with its types.
    (r"py:class", r"wp\.\w+"),
    # The spectrum-source protocols of measure_bf_parallax are private.
    (r"py:class", r"(?:Chunked)?SpectrumSource"),
    # Private modules and members are not documented.
    (r"py:.*", r"(?:[\w.]+\.)?_[A-Za-z0-9_]+.*"),
]

# -- Autodoc --------------------------------------------------------------------

autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_default_options: dict[str, object] = {}
autosummary_generate = False
# GPAW is optional and imported lazily; mocking it changes nothing today and
# keeps a later module-level import from breaking the CPU build.
autodoc_mock_imports = ["gpaw"]

# The docstrings follow the numpydoc standard; a few older ones use the Google
# sections. Napoleon reads both, as in the development documentation.
napoleon_numpy_docstring = True
napoleon_google_docstring = True
# An ``Attributes`` section becomes an ``:ivar:`` list. As ``.. attribute::``
# entries, a dataclass field that is also annotated would be described twice.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST -----------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_block"]
myst_heading_anchors = 3

# -- HTML -------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "scatterem"
html_baseurl = "https://scatterem.github.io/scatterem/"
html_static_path = ["_static"]
html_css_files = ["scatterem.css"]
html_logo = "../logo.png"
html_favicon = "../favicon.png"
# The former site served the logo and the favicon at its root; keep those URLs.
html_extra_path = ["../logo.png", "../favicon.png"]
# The navigation bar lists every page; each page is a top-level page, so a
# left sidebar would only show an empty "Section navigation".
html_sidebars: dict[str, list[str]] = {"**": []}
html_copy_source = False
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/scatterem/scatterem",
    "logo": {"text": "scatterem", "alt_text": "scatterem"},
    "navigation_with_keys": False,
    "show_prev_next": True,
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
# Dark first, as on the former site; the navbar switch changes it.
html_context = {"default_mode": "dark"}

linkcheck_anchors = True
linkcheck_timeout = 20

# -- Completeness of the API reference --------------------------------------------

#: Internal modules. Exported modules import them, so they ship, but they are
#: not public API: the reference must not show them, and their ``__all__``
#: need not be covered. Their names can change without notice.
INTERNAL_MODULES = (
    "scatterem.data.storage",
    "scatterem.utils.env",
    "scatterem.utils.lazy_plotting",
    "scatterem.utils.warp",
)


def _is_internal(name: str) -> bool:
    parts = name.split(".")
    if any(p.startswith("_") and not p.startswith("__") for p in parts):
        return True
    return any(name == m or name.startswith(m + ".") for m in INTERNAL_MODULES)


def _public_modules() -> list[str]:
    """Every public module of the package that has an ``__all__``."""
    names = []
    for info in pkgutil.walk_packages(scatterem.__path__, "scatterem."):
        if _is_internal(info.name) or "tests" in info.name.split("."):
            continue
        module = importlib.import_module(info.name)
        if getattr(module, "__all__", None):
            names.append(info.name)
    return names


def _resolve(dotted: str):
    """The object that a documented dotted name refers to, or None.

    The walk goes attribute by attribute from the package, as Python resolves
    ``scatterem.vis.show`` (a function that the package ``scatterem.vis``
    binds, not the subpackage of the same name). It imports a submodule only
    where no attribute of that name exists.
    """
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, attr in enumerate(parts[1:], start=1):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            continue
        try:
            obj = importlib.import_module(".".join(parts[: i + 1]))
        except ImportError:
            return None
    return obj


def check_api_reference(app, env) -> None:
    """Warn for each public name the reference misses and each internal one it shows."""
    from sphinx.util import logging

    logger = logging.getLogger("api_reference")
    documented = {
        name: obj.docname
        for name, obj in env.get_domain("py").objects.items()
        # An aliased entry is the defining module of a re-exported object
        # (autodoc's ``canonical``); the page shows the public name only.
        if obj.objtype != "module" and not obj.aliased
    }
    if not documented:
        return
    for name in sorted(documented):
        if _is_internal(name):
            logger.warning(
                "the API reference shows %s, which is internal",
                name,
                type="scatterem",
                subtype="api",
            )
    shown_ids = {id(o) for o in map(_resolve, documented) if o is not None}
    modules = dict.fromkeys(["scatterem.vis", *_public_modules()])
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            dotted = f"{module_name}.{name}"
            obj = getattr(module, name, None)
            if dotted in documented:
                continue
            if (callable(obj) or isinstance(obj, type)) and id(obj) in shown_ids:
                continue
            logger.warning(
                "%s is in __all__ but not in docs/api.md; add it, or remove it "
                "from __all__",
                dotted,
                type="scatterem",
                subtype="api",
            )


# -- References to objects that exist but are not on the page ------------------------

#: ``docstring of <dotted name>`` in the source of a docstring node.
_DOCSTRING_OF = re.compile(r"docstring of ((?:scatterem)(?:\.\w+)+)")


def _contexts(node) -> list[str]:
    """Dotted scopes in which a reference in ``node`` may resolve, nearest first."""
    scopes = []
    module, cls = node.get("py:module"), node.get("py:class")
    if module and cls:
        scopes.append(f"{module}.{cls}")
    if module:
        scopes.append(module)
    hit = _DOCSTRING_OF.search(node.source or "")
    if hit:
        # The object whose docstring holds the reference, and each scope above it
        # (its class, its defining module, the packages).
        parts = hit.group(1).split(".")
        scopes += [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
    return list(dict.fromkeys(scopes))


def accept_existing_objects(app, env, node, contnode):
    """Resolve a Python reference to an object that exists but has no entry.

    The reference shows a chosen part of the API (docs/api.md), so a docstring
    can name a method, an attribute or a helper that the page does not show.
    Such a reference renders as code without a link. A reference to a name
    that does not exist in the package (a rename, a removal) is not resolved
    here and stays a warning, so the strict build still fails on it.
    """
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"].lstrip(".~")
    if not target or not target.replace(".", "").replace("_", "").isalnum():
        return None
    if "." not in target and app.builder is not None:
        # An unqualified name (an annotation such as ``SMeta``) links to the
        # one documented object of that name, if there is exactly one.
        py = env.get_domain("py")
        hits = [
            name
            for name, obj in py.objects.items()
            if name.endswith("." + target) and not obj.aliased
        ]
        if len(hits) == 1:
            resolved = py.resolve_xref(
                env,
                node.get("refdoc", ""),
                app.builder,
                node["reftype"],
                hits[0],
                node,
                contnode,
            )
            if resolved is not None:
                return resolved
    names = [target] if target.startswith("scatterem.") else []
    names += [f"{scope}.{target}" for scope in _contexts(node)]
    for name in names:
        if name.startswith("scatterem.") and _exists(name):
            return contnode
    return None


def _exists(dotted: str) -> bool:
    """Whether ``dotted`` names an object or an annotated class field."""
    if _resolve(dotted) is not None:
        return True
    owner, _, field = dotted.rpartition(".")
    cls = _resolve(owner) if owner else None
    return isinstance(cls, type) and any(
        field in getattr(c, "__annotations__", {}) for c in cls.__mro__
    )


# -- Plain docstrings of constants -----------------------------------------------------

#: The docstrings of module constants are prose. Napoleon would read their
#: first line as ``type: description`` and make a type of the words before
#: the first colon, so the original lines are put back after it ran.
_PLAIN: dict[str, list[str]] = {}


def _keep_plain(app, what, name, obj, options, lines):
    if what in {"data", "attribute"}:
        _PLAIN[name] = list(lines)


def _restore_plain(app, what, name, obj, options, lines):
    if name in _PLAIN:
        lines[:] = _PLAIN.pop(name)


# -- Source maps of the theme --------------------------------------------------------


def drop_source_maps(app, exception) -> None:
    """Delete the ``*.map`` files that the theme copies into ``_static``.

    They are the theme's unminified sources, for browser developer tools. The
    site does not need them, and they carry the upstream projects' own issue
    and branch references.
    """
    if exception is None and app.builder.format == "html":
        for path in Path(app.outdir, "_static").rglob("*.map"):
            path.unlink()


def setup(app):
    app.connect("env-check-consistency", check_api_reference)
    app.connect("missing-reference", accept_existing_objects)
    app.connect("build-finished", drop_source_maps)
    # Napoleon connects at the default priority, 500.
    app.connect("autodoc-process-docstring", _keep_plain, priority=400)
    app.connect("autodoc-process-docstring", _restore_plain, priority=600)
