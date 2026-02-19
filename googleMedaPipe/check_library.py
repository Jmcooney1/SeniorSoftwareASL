import numpy as np

# Load the library
try:
    asl_library = np.load('asl_library.npy', allow_pickle=True).item()
    
    print("--- ASL LIBRARY CONTENTS ---")
    print(f"Total letters saved: {len(asl_library)}")
    print("-" * 30)
    
    # Sort them alphabetically for a cleaner look
    for letter in sorted(asl_library.keys()):
        data_points = asl_library[letter].shape[0]
        print(f"Letter: {letter.upper()} | Points: {data_points} (63 is perfect)")

except FileNotFoundError:
    print("❌ Error: 'asl_library.npy' does not exist yet. Run create_library.py first!")
except Exception as e:
    print(f"❌ Error loading file: {e}")