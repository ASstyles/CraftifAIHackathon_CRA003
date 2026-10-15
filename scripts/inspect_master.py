import glob
import subprocess
import time

p = subprocess.Popen(["/opt/PipeGen/PipeGenApp", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
bins = glob.glob("/tmp/onefile_*/PipeGenApp.bin")
if bins:
    with open(bins[0], "rb") as f:
        data = f.read()
    idx = data.find(b"MASTER_B64")
    if idx != -1:
        snippet = data[idx-100:idx+300]
        print("".join(chr(b) if 32 <= b <= 126 else " " for b in snippet))
p.terminate()
