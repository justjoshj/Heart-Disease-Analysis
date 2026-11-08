import subprocess
import sys

libraries = {
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn"
}

for library, package in libraries.items():
    try:
        __import__(library)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package
        ])

print("All libraries installed")