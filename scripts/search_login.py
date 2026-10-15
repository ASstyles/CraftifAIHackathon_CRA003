import re
import glob
import subprocess
import time

p = subprocess.Popen(["/opt/PipeGen/PipeGenApp", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
bins = glob.glob("/tmp/onefile_*/PipeGenApp.bin")
if bins:
    with open(bins[0], "rb") as f:
        data = f.read()
    targets = [b"/auth", b"/login", b"signin", b"dev.craftifai.com", b"TOKEN_FILE"]
    for t in targets:
        for m in re.finditer(t, data):
            idx = m.start()
            snippet = data[max(0, idx-60):min(len(data), idx+140)]
            print(f"[{t.decode()}] " + "".join(chr(b) if 32 <= b <= 126 else " " for b in snippet))
p.terminate()
