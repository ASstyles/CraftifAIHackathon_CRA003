import os
import time
import subprocess

token_path = os.path.expanduser("/home/aviral/.pipegen/token")
default_user_path = os.path.expanduser("/home/aviral/.pipegen/DefaultUser")

print("Monitoring PipeGen authentication state...")
for i in range(1, 16):
    if os.path.exists(token_path):
        print(f"SUCCESS: Token detected at {token_path}!")
        user = open(default_user_path).read().strip() if os.path.exists(default_user_path) else "unknown"
        print(f"Authenticated user: {user}")
        break

    # Check window title
    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        res = subprocess.run(["xdotool", "search", "--name", "PipeGen"], capture_output=True, text=True, env=env)
        win_ids = res.stdout.strip().split()
        names = []
        for wid in win_ids:
            r = subprocess.run(["xdotool", "getwindowname", wid], capture_output=True, text=True, env=env)
            names.append(f"{wid}: {r.stdout.strip()}")
        print(f"[{i}/15] Active windows: {', '.join(names)}")
    except Exception as e:
        print(f"[{i}/15] Checking... ({e})")
    time.sleep(2)
