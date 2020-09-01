#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom errors.
"""


class CPIObjectDoesNotExist(Exception):
    """
    Error raised when a CPI object is requested that doesn't exist.
    """

    pass


class StaleDataWarning(Warning):
    """
    The warning to raise when the local data are out of date.
    """

    def __str__(self):
        return "CPI data is out of date. To accurately inflate to today's dollars, you must run `cpi.update()`."
