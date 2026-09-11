"""Explicitly freeze a NEW reference version before tests; never touch old runs."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'reference').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
table='a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc'
assert hashes['reference/inputs/profile.npz']==table
(root/'reference_manifest.json').write_text(json.dumps(dict(purpose='strict_instruction_interface_v1',input_hashes=hashes),indent=2)+'\n')
