
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

print("Starting import test...")

try:
    from models.phrasal_mcan.iterative_mcan_phrasal import IterativeMCANPhrasal
    print("IterativeMCANPhrasal imported successfully.")
except Exception as e:
    print(f"Error importing IterativeMCANPhrasal: {e}")
    sys.exit(1)

try:
    from builders.model_builder import META_ARCHITECTURE
    if "IterativeMCANPhrasal" in META_ARCHITECTURE:
        print("IterativeMCANPhrasal registered in META_ARCHITECTURE.")
    else:
        print("ERROR: IterativeMCANPhrasal NOT registered.")
except Exception as e:
    print(f"Error checking META_ARCHITECTURE: {e}")

try:
    from builders.encoder_builder import META_ENCODER
    # Check if PhrasalEncoder is registered (it should be imported by the new file)
    if "PhrasalEncoder" in META_ENCODER:
        print("PhrasalEncoder registered in META_ENCODER.")
    else:
        print("ERROR: PhrasalEncoder NOT registered.")
except Exception as e:
    print(f"Error checking META_ENCODER: {e}")
