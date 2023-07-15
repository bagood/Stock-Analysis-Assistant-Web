import sys
import subprocess

class subprocess_functions:
    def __init__(self):
        pass

    def install_packages(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'datetime'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'pytz'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'plotly'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'yfinance'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'scikit-learn'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'keras'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow'])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])