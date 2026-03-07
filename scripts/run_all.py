
import os
import subprocess
import sys
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "fetch_prices.py",
    "fetch_calendar.py",
    "fetch_news.py",
    "backtest_model.py",
    "generate_report.py"
]

# Hardcoded python path since we know 3.12 is the one with packages
#/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Or we can just use "python3.12" if in path.
# Let's try to detect or use the hardcoded one if sys.executable is 3.9
# But the safer way for the USER's machine is to assume 'python3' might be 3.12 if they run it differently, 
# OR just force the one we know works.
PYTHON_EXEC = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

if not os.path.exists(PYTHON_EXEC):
    # Fallback to sys.executable if the hardcoded path doesn't exist
    PYTHON_EXEC = sys.executable

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"--- Running {script_name} ---")
    try:
        # Run using the specific python executable
        result = subprocess.run([PYTHON_EXEC, script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")

def main():
    print(f"[{datetime.now()}] Starting Metals Macro System Update...")
    print(f"Using Python: {PYTHON_EXEC}")
    
    for script in SCRIPTS:
        run_script(script)
        
    print("All tasks completed.")

if __name__ == "__main__":
    main()
