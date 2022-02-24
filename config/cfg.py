#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config directory to access config files across different modules
in order to avoid clashes between experiments
"""
config_directory = None

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self