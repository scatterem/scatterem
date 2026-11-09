"""Which colour maps scatterem allows, as plain data.

A CI gate runs this file with ``runpy.run_path`` and does not import
``scatterem``. Keep this module free of import statements.

The allow-list follows check 4 of Crameri, Shephard & Heron (2020), "The misuse
of colour in science communication", Nat. Commun. 11, 5444: use a colour map
only if it is scientifically derived. scatterem allows the Scientific Colour
Maps, the matplotlib perceptual maps (Box 2 of the paper) and two greys.
Colorbrewer is in Box 2 too, but it ships only discrete maps, so it is left out.
"""

#: Every base name in cmcrameri 1.10 (Scientific Colour Maps 8.0.1).
CRAMERI = (
    "acton", "actonS", "bam", "bamO", "bamako", "bamakoS", "batlow", "batlowK",
    "batlowKS", "batlowS", "batlowW", "batlowWS", "berlin", "bilbao", "bilbaoS",
    "broc", "brocO", "buda", "budaS", "bukavu", "cork", "corkO", "davos",
    "davosS", "devon", "devonS", "fes", "glasgow", "glasgowS", "grayC", "grayCS",
    "hawaii", "hawaiiS", "imola", "imolaS", "lajolla", "lajollaS", "lapaz",
    "lapazS", "lipari", "lipariS", "lisbon", "managua", "navia", "naviaS", "nuuk",
    "nuukS", "oleron", "oslo", "osloS", "roma", "romaO", "tofino", "tokyo",
    "tokyoS", "turku", "turkuS", "vanimo", "vik", "vikO",
)  # fmt: skip

#: The matplotlib maps that Box 2 of the paper lists as scientifically derived.
MATPLOTLIB = ("viridis", "plasma", "inferno", "magma", "cividis")

#: Greys. grayC is perceptually uniform; gray stays for interoperability.
GREYS = ("gray", "grayC")

#: Maps the policy hands out, with the Fig. 6 class each one is tested as.
POLICY_MAPS = {
    "batlow": "sequential",
    "batlow_r": "sequential",
    "batlowK": "sequential",
    "grayC": "sequential",
    "lipari": "sequential",
    "devon": "sequential",
    "lajolla": "sequential",
    "berlin": "diverging",
    "vik": "diverging",
    "oleron": "multi-sequential",
    "bukavu": "multi-sequential",
    "romaO": "cyclic",
}

#: Density maps offered by a report; the first one is the default.
DENSITY_MENU = ("batlow", "grayC", "batlowK", "lipari")

#: Maps offered by a dark-themed viewer. The viewer is dark, so vik comes before berlin.
GUI_MENU = ("batlow", "batlowK", "grayC", "lipari", "vik", "berlin", "romaO")

#: The role to suggest when the gate or the runtime warning meets a banned map.
SUGGESTIONS = {
    "jet": "intensity",
    "rainbow": "intensity",
    "gist_rainbow": "intensity",
    "nipy_spectral": "intensity",
    "gist_ncar": "intensity",
    "turbo": "intensity",
    "hot": "intensity",
    "Greys": "intensity",
    "Grays": "intensity",
    "hsv": "angle",
    "twilight": "angle",
    "twilight_shifted": "angle",
    "coolwarm": "signed",
    "bwr": "signed",
    "seismic": "signed",
    "RdBu": "signed",
    "Spectral": "signed",
    "RdYlGn": "signed",
    "RdYlBu": "signed",
    "Blues": "species_a",
    "Reds": "species_b",
}


def is_allowed(name: str) -> bool:
    """Whether ``name`` is a scientifically derived map scatterem accepts.

    Parameters
    ----------
    name : str
        A matplotlib colormap name, with or without the ``cmc.`` prefix or the
        ``_r`` suffix.

    Returns
    -------
    bool
        True for Crameri, matplotlib-perceptual and grey maps.
    """
    base = name.removeprefix("cmc.").removesuffix("_r")
    return base in CRAMERI or base in MATPLOTLIB or base in GREYS


def suggestion_for(name: str) -> str:
    """The role to use in place of a banned map.

    Parameters
    ----------
    name : str
        The banned colormap name.

    Returns
    -------
    str
        A key of ``scatterem.vis.colors.ROLES``; ``"intensity"`` when unknown.
    """
    return SUGGESTIONS.get(name.removesuffix("_r"), "intensity")
