from __future__ import absolute_import
from pyshortcuts import make_shortcut
import zeclinics_app_tox as z
import os
make_shortcut(os.path.abspath(z.__file__), name = 'Zeclinics', desktop=True, terminal=True)
