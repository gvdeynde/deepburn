# -*- coding: utf-8 -*-
import re

SYMBOLS = [
    "n",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

ELEMENTS = [
    "Neutron",
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminum",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
    "Germanium",
    "Arsenic",
    "Selenium",
    "Bromine",
    "Krypton",
    "Rubidium",
    "Strontium",
    "Yttrium",
    "Zirconium",
    "Niobium",
    "Molybdenum",
    "Technetium",
    "Ruthenium",
    "Rhodium",
    "Palladium",
    "Silver",
    "Cadmium",
    "Indium",
    "Tin",
    "Antimony",
    "Tellurium",
    "Iodine",
    "Xenon",
    "Cesium",
    "Barium",
    "Lanthanum",
    "Cerium",
    "Praseodymium",
    "Neodymium",
    "Promethium",
    "Samarium",
    "Europium",
    "Gadolinium",
    "Terbium",
    "Dysprosium",
    "Holmium",
    "Erbium",
    "Thulium",
    "Ytterbium",
    "Lutetium",
    "Hafnium",
    "Tantalum",
    "Tungsten",
    "Rhenium",
    "Osmium",
    "Iridium",
    "Platinum",
    "Gold",
    "Mercury",
    "Thallium",
    "Lead",
    "Bismuth",
    "Polonium",
    "Astatine",
    "Radon",
    "Francium",
    "Radium",
    "Actinium",
    "Thorium",
    "Protactinium",
    "Uranium",
    "Neptunium",
    "Plutonium",
    "Americium",
    "Curium",
    "Berkelium",
    "Californium",
    "Einsteinium",
    "Fermium",
    "Mendelevium",
    "Nobelium",
    "Lawrencium",
    "Rutherfordium",
    "Dubnium",
    "Seaborgium",
    "Bohrium",
    "Hassium",
    "Meitnerium",
    "Darmstadtium",
    "Roentgenium",
    "Copernicium",
    "Nihonium",
    "Flerovium",
    "Moscovium",
    "Livermorium",
    "Tennessine",
    "Oganesson",
]

RE_AAAMZZZ = re.compile(r"([0-9]*)([m]*)([+\-_\*/]*)([A-Z][a-z]*)", re.UNICODE)
RE_ZZZAAAM = re.compile(r"([A-Z][a-z]*)([+\-_\*/]*)([0-9]*)([m]*)", re.UNICODE)


def _zzzaaam2str(zzzaaam, symbol=True, zfirst=True, meta=True, separator="-"):
    """Conversion function to return string from ZZZAAAM format
        Returned string can be full name or symbol.

    Args:
        zzzaaam (int or string): ZZZAAAM notation

        symbol (bool, optional): if True, returns the symbol. If False,
        returns the full name. Default: 'True'

        zfirst (bool, optional): if True, the proton name is first, followed by
        the massname (Am241); if False, order is reversed (241Am)

        meta (bool, optional): if True, the metastable state is added

        separator (string, optional): seperator between element name/symbol
        and mass number. Default: '-'

    Returns:
        name: the name

    Raises:
        TypeError: if zzzaaam is not a string or an integer
        ValueError: if zzzaaam is non-conformant
    """

    if not isinstance(zzzaaam, str):
        zzzaaam = str(zzzaaam)

    if len(zzzaaam) < 4 or len(zzzaaam) > 7:
        raise ValueError(
            f"Length of zzzaaam should be between 4 and 7, not {len(zzzaaam)}"
        )

    mass_number = int(zzzaaam[-4:-1])
    proton_number = int(zzzaaam[:-4])
    metastable_state = int(zzzaaam[-1])

    if symbol:
        protonname = SYMBOLS[proton_number]
    else:
        protonname = ELEMENTS[proton_number]

    massname = str(mass_number)

    if meta:
        metaname = metastable_state * "m"
    else:
        metaname = ""

    if zfirst:
        name = protonname + separator + massname + metaname
    else:
        name = massname + metaname + separator + protonname

    return name


def _str2zzzaaam(name):
    """Conversion function to return a tuple (zzz, aaaa, m) from a string.
       The function tries to be as intelligent as possible

    Args:
        name (string): the name of the isotope. Can be Am241m, Americium-241m,
        241mAm, 241m-Americium

    Returns
        zzzaaam (tuple): tuple containing (zzz, aaa, m) as integers
    """

    # Check first character of name. If it's a number, we assume AAAMZZZ format

    if name[0].isnumeric():
        match = RE_AAAMZZZ.match(name)
        aaa = int(match[1])
        m = len(match[2])

        # If length of zzz is 1 or 2, we assume it is a symbol
        if len(match[4]) == 1 or len(match[4]) == 2:
            try:
                zzz = int(SYMBOLS.index(match[4]))
            except ValueError:
                raise ValueError(f"I don't know element {match[4]}")
        else:
            try:
                zzz = int(ELEMENTS.index(match[4].title()))
            except ValueError:
                raise ValueError(f"I don't know element {match[4]}")

    else:
        match = RE_ZZZAAAM.match(name)
        aaa = int(match[3])
        m = len(match[4])

        # If length of zzz is 1 or 2, we assume it is a symbol
        if len(match[1]) == 1 or len(match[1]) == 2:
            try:
                zzz = int(SYMBOLS.index(match[1]))
            except ValueError:
                raise ValueError(f"I don't know element {match[1]}")
        else:
            try:
                zzz = int(ELEMENTS.index(match[1].title()))
            except ValueError:
                raise ValueError(f"I don't know element {match[1]}")

    return (zzz, aaa, m)

class Isotope:
    """Class to hold isotope and pretty-printing"""

    def __init__(self, iso):
        if isinstance(iso, str):
            iso = _str2zzzaaam(iso)
        else:
            if len(iso) < 2:
                raise ValueError("I need at least proton and mass number")
            if len(iso) < 3:
                iso = (iso[0], iso[1], 0)

        self._zzz = iso[0]
        self._aaa = iso[1]
        self._meta = iso[2]

    def __format__(self, code):
        if code == '':
            code = 'Sam'
        
        if code == 'zam':
            res = f"{self._zzz}-{self._aaa}"
        elif code == 'Eam':
            res = f"{ELEMENTS[self._zzz]}-{self._aaa}"
        elif code == 'Sam':
            res = f"{SYMBOLS[self._zzz]}-{self._aaa}"
        else:
            raise ValueError("wrong format code")

        if self._meta != 0:
            res += 'm'
        return res

    def __str__(self):
        return f"{self}"

    @property
    def zzz(self):
        """ Returns the proton number """
        return self._zzz

    @zzz.setter
    def zzz(self, value):
        """ Read-only property """
        raise ValueError("'zzz' is a read-only property")

    @property
    def aaa(self):
        """ Returns the proton number """
        return self._aaa

    @aaa.setter
    def aaa(self, value):
        """ Read-only property """
        raise ValueError("'aaa' is a read-only property")

    @property
    def meta(self):
        """ Returns the proton number """
        return self._meta

    @meta.setter
    def meta(self, value):
        """ Read-only property """
        raise ValueError("'meta' is a read-only property")

    @property
    def name(self):
        zzzaaam = f"{self._zzz:3d}{self._aaa:3d}{self._meta:1d}"
        return _zzzaaam2str(zzzaaam)

    @name.setter
    def name(self, value):
        """ Read-only property """
        raise ValueError("'name' is a read-only property")
