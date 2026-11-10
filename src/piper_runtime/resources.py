"""Read-only packaged Piper kinematic XML and exported clearance geometry."""
from pathlib import Path
DATA=Path(__file__).with_name('data')
MODEL_XML=DATA/'piper.xml'
GEOMETRY=DATA/'safety-geometry.npz'
