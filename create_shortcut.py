from pyshortcuts import make_shortcut
import os
print(os.listdir('.'))
os.rename('zeclinics_app','zeclinics_app_2')
os.rename('terato','terato_2')
os.rename('cardio','cardio_2')
import zeclinics_app as z
print(os.path.abspath(z.__file__))
make_shortcut(os.path.abspath(z.__file__), name = 'Zeclinics', desktop=True, terminal=True)
os.rename('zeclinics_app_2','zeclinics_app')
os.rename('terato_2','terato')
os.rename('cardio_2','cardio')
