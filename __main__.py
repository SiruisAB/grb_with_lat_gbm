# -*- coding: utf-8 -*-
"""Command-line entry point for ``python -m grb_project``."""

import sys

from .pipeline import cli_main


_rc = cli_main()
sys.exit(_rc if isinstance(_rc, int) else 0)
