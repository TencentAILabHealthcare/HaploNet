# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Constants and helpers for MSA construction."""

from collections import OrderedDict, namedtuple
from functools import lru_cache

BASE28_TYPES = (
    "A","C","G","T","N","D","I","a","c","g","t","n","d","i",
    "A+","C+","G+","T+","N+","a+","c+","g+","t+","n+",
    "M","m","M+","m+",
)

IUPAC_TO_BASE = {
    'R':('A','G'),'Y':('C','T'),'S':('G','C'),
    'W':('A','T'),'K':('G','T'),'M':('A','C'),
    'B':('C','G','T'),'D':('A','G','T'),
    'H':('A','C','T'),'V':('A','C','G'),
}


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)


Candidate = namedtuple("Candidate", ["chrom", "pos", "beta"])


@lru_cache(maxsize=1000)
def check_base(base):
    """Validate and normalize a single base character to uppercase."""
    na = base.upper()
    if na not in (".","-","*","#","A","G","C","T","N","U"):
        na = IUPAC_TO_BASE.get(na, (["N"],))[0][0] if na in IUPAC_TO_BASE else "N"
    return na


def check_seq(seq):
    return "".join(check_base(b) for b in seq)


def is_interval_in(tree, start=None, end=None):
    if hasattr(tree, 'at'):
        return len(tree.at(start) if end is None else tree.overlap(begin=start, end=end)) > 0
    return len(tree.search(begin=start, end=end, strict=False)) > 0


def parse_region(region_str):
    """Parse genomic region string 'chr:start-end' into (chrom, start, end)."""
    try:
        cp, pp = region_str.split(":")
        ss, es = pp.split("-")
        chrom = cp.strip()
        start = int(ss.strip())
        end = int(es.strip())
        if end <= start:
            raise ValueError
        return chrom, start, end
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid region '{region_str}', expected chr:start-end ({e})")

# Lazy import to avoid circular dependency at module level
import argparse as _argparse
argparse = _argparse
