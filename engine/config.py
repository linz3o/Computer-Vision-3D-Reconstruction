"""
import json

with open('config.json') as file:
    config = json.load(file)
"""

import json
import os

# Define the file paths
default_config_path = 'config.json'
fallback_config_path = 'C:/Users/linhd/Computer-Vision-1/Computer-Vision-3D-Reconstruction/config.json'

# Try reading from the default path first
if os.path.exists(default_config_path):
    with open(default_config_path) as file:
        config = json.load(file)
else:
    # If the default config doesn't exist, read from the fallback path
    with open(fallback_config_path) as file:
        config = json.load(file)