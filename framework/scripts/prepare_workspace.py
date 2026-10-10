"""Materialize the archived experiment paths into an empty local workspace."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(repository, workspace, qwen25_python, vlm_python):
    repository, workspace = Path(repository).resolve(), Path(workspace).resolve()
    manifest = json.loads((repository / 'framework/configs/source_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        path = repository / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['published_sha256']:
            raise ValueError(f'Published source hash mismatch: {entry["path"]}')
        destination = workspace / entry['source']
        if workspace not in destination.resolve().parents:
            raise ValueError(f'Invalid destination: {entry["source"]}')
        if destination.exists():
            expected = path.read_text(encoding='utf-8').replace('@@WORKSPACE@@',str(workspace))
            expected = expected.replace('@@QWEN25_PYTHON@@',qwen25_python).replace('@@VLM_PYTHON@@',vlm_python)
            expected = expected.replace('@@SERVER_ROOT@@',str(workspace.parent))
            expected = expected.replace('@@HOME@@',str(Path.home()))
            if destination.read_text(encoding='utf-8') != expected:
                raise FileExistsError(f'Existing file differs: {destination}')
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        text=path.read_text(encoding='utf-8').replace('@@WORKSPACE@@',str(workspace))
        text=text.replace('@@QWEN25_PYTHON@@',qwen25_python).replace('@@VLM_PYTHON@@',vlm_python)
        text=text.replace('@@SERVER_ROOT@@',str(workspace.parent))
        destination.write_text(text.replace('@@HOME@@',str(Path.home())),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--qwen25-python',default='python')
    parser.add_argument('--vlm-python',default='python')
    args=parser.parse_args()
    prepare(Path(__file__).resolve().parents[2],args.workspace,args.qwen25_python,args.vlm_python)
