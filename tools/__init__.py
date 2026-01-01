#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SatUpdate Tools Package

This package provides utility tools for the satellite constellation
simulator, including:

- SPK Export: Export constellation ephemerides to SPICE SPK format
"""

from .generate_spk import (
    SPKGenerator,
    SPKSegment,
    StateVector,
    NAIFIDManager,
    create_spk_from_simulation,
    NAIF_ID_BASE,
    NAIF_ID_MIN,
    NAIF_ID_MAX,
    EARTH_NAIF_ID,
)

__all__ = [
    "SPKGenerator",
    "SPKSegment",
    "StateVector",
    "NAIFIDManager",
    "create_spk_from_simulation",
    "NAIF_ID_BASE",
    "NAIF_ID_MIN",
    "NAIF_ID_MAX",
    "EARTH_NAIF_ID",
]

__version__ = "1.0.0"