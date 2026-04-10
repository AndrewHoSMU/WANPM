# Sequential runner for training scripts
# Run with: python run_sequential.py

import subprocess
import sys
import os
from pathlib import Path

PYTHON_EXE = r"D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe"
BASE_DIR = Path(r"D:\PythonProjects\WANPM_FP\Fractional")

SCRIPTS = [
    ("fFP_1d_1well", "fFP_1d_1well.py"),
    ("fFP_1d_2well", "fFP_1d_2well.py"),
    ("fFP_1d_steady_doublepeak", "fFP_1d_steady_doublepeak.py"),
    ("fFP_1d_steady_ou", "fFP_1d_steady_ou.py"),
    ("fFP_1d_triplewell", "fFP_1d_triplewell.py"),
    ("fFP_20d_doublewell", "fFP_20d_doublewell.py"),
    ("fFP_2d_ring", "fFP_2d_ring.py"),
    ("fFP_2d_steady_doublepeak", "fFP_2d_steady_doublepeak.py"),
    ("fFP_2d_steady_ring", "fFP_2d_steady_ring.py"),
    ("fFP_nd_ou", "fFP_nd_ou.py"),
    ("fFP_nd_1well", "FP_nd_1well.py"),
]

def run_script(script_dir: str, script_file: str, index: int, total: int) -> bool:
    """Run a single script and return True if successful."""
    work_dir = BASE_DIR / script_dir
    log_file = work_dir / "run.log"
    
    print(f"\n[{index}/{total}] Starting: {script_file}")
    print(f"  Working dir: {work_dir}")
    print(f"  Log: {log_file}")
    
    # Set environment variable for UTF-8 encoding
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                [PYTHON_EXE, script_file],
                cwd=work_dir,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if result.returncode == 0:
            print(f"[{index}/{total}] DONE: {script_file}")
            return True
        else:
            print(f"[{index}/{total}] ERROR (exit {result.returncode}): {script_file}")
            return False
    except Exception as e:
        print(f"[{index}/{total}] FAILED: {e}")
        return False

def main():
    print(f"Sequential Training Runner")
    print(f"Python: {PYTHON_EXE}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Total scripts: {len(SCRIPTS)}")
    
    success_count = 0
    fail_count = 0
    
    for i, (script_dir, script_file) in enumerate(SCRIPTS, 1):
        if run_script(script_dir, script_file, i, len(SCRIPTS)):
            success_count += 1
        else:
            fail_count += 1
        print("-" * 60)
    
    print(f"\nAll done!")
    print(f"Success: {success_count}, Fail: {fail_count}")

if __name__ == "__main__":
    main()
