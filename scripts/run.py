"""Open the browser only after the local service is listening."""
from pathlib import Path
import sys
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def open_when_ready():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        try:
            with opener.open('http://127.0.0.1:8765/api/health', timeout=1) as response:
                if response.status == 200:
                    webbrowser.open('http://127.0.0.1:8765')
                    return
        except OSError:
            pass
        time.sleep(0.2)


if __name__ == '__main__':
    import uvicorn
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run('backend.app:app', host='127.0.0.1', port=8765, access_log=False)
