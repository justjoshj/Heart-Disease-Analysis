import subprocess
import sys

subprocess.run([sys.executable, "models/confusion_matrix.py"])
subprocess.run([sys.executable, "models/correlation_heatmap.py"])
subprocess.run([sys.executable, "models/feature_importance.py"])