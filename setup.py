from setuptools import setup, find_packages

setup(
    name="zeclinics_app",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "zeclinicsapp = zeclinics_app:start"
        ]
    },
    install_requires=[
        "flask",
        "werkzeug",
        "Pillow",
        "torch",
        "read_roi",
        "opencv-python",
        "imageio",
        "scipy",
        "torchvision",
        "matplotlib",
        "numpy",
        "pathlib",
        "heartpy",
        "flaskwebgui",
        "tqdm",
        "importlib_resources",
        "importlib",
        "readlif",
        "altair",
        "bioinfokit",
        "prince"
    ],
    package_data={
        'zeclinics_app': ['templates/*','static/css/*', 'static/images/*', 'static/img/*', 'static/js/*', 'static/videos/*', 'static/weight/*'],
    },
)

'''
package_data={
    'static': ['css/*', 'images/*', 'img/*', 'js/*', 'videos/*', 'weight/*'],
    'templates': ['*'],

},
'''
