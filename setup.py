from setuptools import setup, find_packages

setup(
    name="zeclinics_app",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "zeclinicsapp = zeclinics_app.app:start"
        ]
    },
    install_requires=[
        "flask",
        "werkzeug",
        "Pillow",
        "torch",
        "read_roi",
        "roifile",
        "scikit-image",
        "opencv-python",
        "imageio",
        "scipy",
        "torchvision",
        "matplotlib<=3.1.2",
        "numpy",
        "pathlib",
        "heartpy==1.2.6",
        "flaskwebgui",
        "tqdm",
        "readlif",
        "altair",
        "bioinfokit",
        "prince",
        "scikit-learn",
        "mpld3",
        "pyshortcuts",
        "gdown"
    ],
    package_data={
        'zeclinics_app': ['templates/*','static/css/*', 'static/images/*', 'static/img/*', 'static/js/*', 'static/videos/*', 'static/weight/*'],
    },
)
