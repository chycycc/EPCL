"""Package a new, versioned tools directory; never overwrite deployed inference tools."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from audit_sources import sha256
from glove_audit import CLOUD_GLOVE_SHA256

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folder', default='training_smoke_tools_20261026_v1')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Use a fresh archive path')
    if Path(args.folder).name != args.folder or args.folder in ('.', '..'):
        parser.error('Expected one directory name')
    files = {p.name: p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md')}
    files.update({'tests/' + p.name: p for p in (HERE / 'tests').glob('*.py')})
    manifest = {'scope': 'bounded_training_smoke_only', 'formal_training_ready': False,
                'weights_included': False, 'data_included': False, 'glove_sha256': CLOUD_GLOVE_SHA256,
                'files_sha256': {name: sha256(path) for name, path in sorted(files.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(args.output), 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(str(path), arcname=args.folder + '/' + name, recursive=False)
        payload = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
        info = tarfile.TarInfo(args.folder + '/bundle_manifest.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    result = {'archive': args.output.name, 'archive_sha256': sha256(args.output),
              'archive_bytes': args.output.stat().st_size, 'folder': args.folder, **manifest}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('archive', 'archive_sha256', 'archive_bytes', 'folder')}))


if __name__ == '__main__':
    main()
