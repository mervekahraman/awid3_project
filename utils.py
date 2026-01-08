import subprocess
import shutil
import os
from typing import List

def ensure_dirs(paths: List[str]):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def which(program: str) -> str:
    """tshark yolu bulur veya None döner"""
    return shutil.which(program)

def run_cmd(cmd: List[str], check=True):
    """Basit subprocess wrapper"""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Komut hata ile döndü: {' '.join(cmd)}\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")
    return res
