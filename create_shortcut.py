from pyshortcuts import make_shortcut
from pathlib import Path
import os
print(os.listdir('.'))
os.rename('zeclinics_app','zeclinics_app_2')
os.rename('terato','terato_2')
os.rename('cardio','cardio_2')
import zeclinics_app as z
print(str(Path(os.path.abspath(z.__file__)).parent/ 'app.py'))
make_shortcut(str(Path(os.path.abspath(z.__file__)).parent/ 'app.py'), name = 'Zeclinics', desktop=True, terminal=True)
os.rename('zeclinics_app_2','zeclinics_app')
os.rename('terato_2','terato')
os.rename('cardio_2','cardio')
