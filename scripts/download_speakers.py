"""Download public sherpa-onnx models only; no audio leaves this computer."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'models' / 'speakers'
BASE = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/'


def download(tag, name, output):
    request = urllib.request.Request(BASE + tag, headers={'User-Agent': 'Tingji'})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    asset = next(a for a in release['assets'] if a['name'] == name)
    temp = output.with_suffix('.part')
    digest = hashlib.sha256()
    request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'Tingji'})
    print(f'Downloading {name} ({asset["size"] / 1024**2:.1f} MB)', flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    if temp.stat().st_size != asset['size']:
        raise RuntimeError('Incomplete model download. Please retry.')
    expected = asset.get('digest') or ''
    if expected.startswith('sha256:') and expected != 'sha256:' + digest.hexdigest():
        raise RuntimeError('Model checksum mismatch. Please retry.')
    temp.replace(output)
    return {'url': asset['browser_download_url'], 'sha256': digest.hexdigest()}


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    sources = {}
    metadata = TARGET / 'sources.json'
    if metadata.exists():
        sources = json.loads(metadata.read_text(encoding='utf-8'))
    if not (TARGET / 'segmentation.onnx').exists():
        archive = TARGET / 'segmentation.tar.bz2'
        sources['segmentation'] = download('speaker-segmentation-models',
            'sherpa-onnx-pyannote-segmentation-3-0.tar.bz2', archive)
        with tarfile.open(archive) as package:
            for original, output in [('model.onnx', 'segmentation.onnx'), ('LICENSE', 'segmentation-LICENSE')]:
                member = next(m for m in package.getmembers() if m.isfile() and Path(m.name).name == original)
                temp = TARGET / (output + '.part')
                with package.extractfile(member) as src, temp.open('wb') as out:
                    shutil.copyfileobj(src, out)
                temp.replace(TARGET / output)
        archive.unlink()
    if not (TARGET / 'embedding.onnx').exists():
        sources['embedding'] = download('speaker-recongition-models',
            '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx', TARGET / 'embedding.onnx')
    metadata.write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print('Speaker models ready. Restart Tingji to use speaker labels.', flush=True)


if __name__ == '__main__':
    main()
