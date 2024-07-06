import subprocess
import threading
import os
import time 

LOCK_FILE = 'server.lock'

def run_app():
    if not os.path.exists(LOCK_FILE):
        subprocess.run(['python', 'app/Balram-server.py'])
    else:
        print("Server is already running.")

def run_streamlit():
    subprocess.run(['streamlit', 'run', 'UI+client/balram-client.py'])

def run():
    # Run each process in a separate thread
    t1 = threading.Thread(target=run_app)
    t2 = threading.Thread(target=run_streamlit)
    t1.start()
    time.sleep(10)
    t2.start()

if __name__ == "__main__":
    run()
