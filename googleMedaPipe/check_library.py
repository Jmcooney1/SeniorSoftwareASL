import numpy as np
import os

library_path = 'asl_library.npy'

if not os.path.exists(library_path):
    print(f"❌ Error: '{library_path}' not found. Run your recorder first!")
else:
    # Load the library
    asl_library = np.load(library_path, allow_pickle=True).item()
    
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    recorded_keys = sorted(asl_library.keys())
    
    print(f"--- ASL LIBRARY STATUS REPORT ---")
    print(f"Total Variations Saved: {len(recorded_keys)}\n")

    missing = []
    
    # Check each letter of the alphabet
    for letter in alphabet:
        # Find all keys that start with this letter (e.g., 'a_left_front', 'a_right_side')
        variations = [k for k in recorded_keys if k.startswith(letter)]
        
        if variations:
            # Join the variations for a clean display
            var_list = ", ".join(variations)
            print(f"✅ {letter.upper()}: {len(variations)} variations found ({var_list})")
        else:
            print(f"❌ {letter.upper()}: MISSING")
            missing.append(letter.upper())

    print("\n" + "="*30)
    if not missing:
        print("🎉 CONGRATS! You have at least one version of every letter.")
    else:
        print(f"⚠️ STILL NEEDED: {', '.join(missing)}")
    print("="*30)