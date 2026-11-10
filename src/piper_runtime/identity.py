"""Version and content identity for locking both consumers to one implementation."""
from pathlib import Path
import argparse
import hashlib
import json
from . import __version__

def identity():
    root=Path(__file__).parent.parent;files={}
    for package in ('piper_runtime','piper_policy_runtime'):
        for path in sorted((root/package).rglob('*')):
            if path.is_file()and path.suffix in ('.py','.cpp','.xml','.npz','.json')and '__pycache__'not in path.parts:
                files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return {'distribution':'piper-runtime','version':__version__,'sha256':digest,'files':files}

def verify(lock_path):
    expected=json.loads(Path(lock_path).read_text());actual=identity()
    if expected['version']!=actual['version']or expected['sha256']!=actual['sha256']:raise ValueError('Installed piper-runtime version/content differs from the consumer lock')
    return actual

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path);args=p.parse_args()
    print(json.dumps(verify(args.lock)if args.lock else identity(),indent=2))
if __name__=='__main__':main()
