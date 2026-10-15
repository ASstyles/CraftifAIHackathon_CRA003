import glob
import subprocess
import time

p = subprocess.Popen(["/opt/PipeGen/PipeGenApp", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
bins = glob.glob("/tmp/onefile_*/PipeGenApp.bin")
if bins:
    with open(bins[0], "rb") as f:
        data = f.read()
    idx = 0
    while True:
        idx = data.find(b"LOGIN_PAGE", idx)
        if idx == -1: break
        snippet = data[max(0, idx-100):min(len(data), idx+200)]
        print("--- SNIPPET ---")
        print("".join(chr(b) if 32 <= b <= 126 else " " for b in snippet))
        idx += len("LOGIN_PAGE")
p.terminate()
