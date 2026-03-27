import numpy as np
import os

static_lib_path = 'asl_library.npy'
motion_lib_path = 'asl_motion_library.npy'

def load_lib(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return {}

static_lib = load_lib(static_lib_path)
motion_lib = load_lib(motion_lib_path)

alphabet = "abcdefghijklmnopqrstuvwxyz"

print("--- FULL ASL PROGRESS REPORT ---")
print(f"Static Signs: {len(static_lib)} | Motion Signs: {len(motion_lib)}\n")

missing = []

for letter in alphabet:
    # Check if the letter exists in either library
    static_vars = [k for k in static_lib.keys() if k.lower().startswith(letter)]
    motion_vars = [k for k in motion_lib.keys() if k.lower().startswith(letter)]
    
    if static_vars or motion_vars:
        status = []
        if static_vars: status.append(f"{len(static_vars)} static")
        if motion_vars: status.append(f"{len(motion_vars)} motion")
        print(f"✅ {letter.upper()}: Found ({', '.join(status)})")
    else:
        print(f"❌ {letter.upper()}: MISSING")
        missing.append(letter.upper())

print("\n" + "="*30)
if not missing:
    print("🎉 FULL ALPHABET COMPLETE!")
else:
    print(f"⚠️ NEEDED: {', '.join(missing)}")