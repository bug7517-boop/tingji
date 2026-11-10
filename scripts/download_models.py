"""Download official model artifacts. No audio or transcript is sent anywhere."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
ROOT = Path(__file__).resolve().parents[1]


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Tingji-Local/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def engine():
    target = ROOT / 'vendor' / 'llama-b6000'
    if target.exists() and next(target.rglob('llama-server.exe'), None):
        print('Summary engine already installed.', flush=True)
        return
    releases = [get_json('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b6000')]
    asset = next((a for release in releases for a in release['assets']
                  if a['name'].endswith('-bin-win-cpu-x64.zip')), None)
    if not asset:
        raise RuntimeError('Official Windows CPU engine was not found. Try again later.')
    print(f"Downloading official llama.cpp: {asset['name']}", flush=True)
    archive = ROOT / 'vendor' / 'llama.zip.part'
    archive.parent.mkdir(exist_ok=True)
    request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'Tingji-Local/1.0'})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    if archive.stat().st_size != asset['size']:
        raise RuntimeError('Engine download is incomplete. Run this script again.')
    expected = asset.get('digest', '')
    if expected.startswith('sha256:') and digest.hexdigest() != expected.split(':', 1)[1]:
        archive.unlink()
        raise RuntimeError('Engine checksum mismatch.')
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            destination = (target / member.filename).resolve()
            if target.resolve() not in destination.parents:
                raise RuntimeError('Unsafe path in engine archive.')
        package.extractall(target)
    archive.unlink()
    (target / 'source.json').write_text(json.dumps({'release': asset['name'], 'url': asset['browser_download_url'],
                                                  'sha256': digest.hexdigest()}, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper', choices=['tiny', 'base', 'small', 'medium'], default='small')
    parser.add_argument('--skip-summary', action='store_true')
    args = parser.parse_args()
    from faster_whisper.utils import download_model
    print(f'Downloading Whisper {args.whisper} (official Systran conversion)...', flush=True)
    download_model(args.whisper, output_dir=str(ROOT / 'models' / 'whisper' / args.whisper))
    if not args.skip_summary:
        engine()
        from huggingface_hub import hf_hub_download
        print('Downloading Qwen2.5 1.5B Q4_K_M (~1.0 GB)...', flush=True)
        hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF',
                        filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir=str(ROOT / 'models'))
    print('Models ready. Start the app and import a recording.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Download failed: {exc}\nRun this script again to resume model downloads.', file=sys.stderr)
        sys.exit(1)
