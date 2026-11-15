"""Keeps one loaded model between calls, so continuing does not pay for imports and loading again. Exits after an idle period.

    engine_server.py SOCKET WEIGHTS [IDLE_S]
One JSON request per connection: {"argv": ops arguments}. Answer: {"returncode", "stdout", "stderr"}."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
path, weights = sys.argv[1], sys.argv[2]
idle = float(sys.argv[3]) if len(sys.argv) > 3 else 300.
os.environ['M3_FAST'] = weights
from noteforge.m3 import draft_m3
from noteforge_api import ops
ops.WARM_DRAFT = draft_m3.run
draft_m3.load(None, None, None)  # load now: the first call should already be fast

Path(path).unlink(missing_ok=True)
server = socket.socket(socket.AF_UNIX)
previous = os.umask(0o177)
server.bind(path)
os.umask(previous)
server.listen(4)
server.settimeout(idle)
try:
    while True:
        try:
            conn, _ = server.accept()
        except TimeoutError:
            break
        with conn:
            conn.settimeout(None)
            data = b''
            while not data.endswith(b'\n'):
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
            if not data:
                continue
            out, err, code = io.StringIO(), io.StringIO(), 0
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = ops.main(json.loads(data)['argv'])
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
                except Exception:  # noqa: BLE001 - one bad call must not end the engine
                    traceback.print_exc()
                    code = 1
            draft_m3.mx.clear_cache()
            conn.sendall((json.dumps(dict(returncode=code, stdout=out.getvalue(), stderr=err.getvalue()))+'\n').encode())
finally:
    Path(path).unlink(missing_ok=True)
    Path(path+'.pid').unlink(missing_ok=True)
