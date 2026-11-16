"""Private subprocess entry point for local speaker analysis."""
import json
from pathlib import Path
import sys

from .speakers import detect_local


def main():
    source, count, folder = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    last = -1
    def progress(ratio):
        nonlocal last
        percent = int(ratio * 100)
        if percent != last:
            (folder / f'progress-{percent:03}.json').write_text(json.dumps(ratio), encoding='utf-8')
            last = percent
    if (folder/'refinement.json').exists():
        from .speaker_refinement import optimize_local
        job=json.loads((folder/'refinement.json').read_text(encoding='utf-8'))
        turns=optimize_local(source,job,progress)
    else:
        turns = detect_local(source, count, progress=progress)
    (folder / 'result.json').write_text(json.dumps(turns), encoding='utf-8')


if __name__ == '__main__':
    main()
