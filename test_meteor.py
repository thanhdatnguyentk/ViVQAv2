import os
import sys
import traceback

# Add path for module
sys.path.append(os.getcwd())

try:
    from evaluation.meteor.meteor import Meteor
    print("--- Meteor module found ---")
except ImportError as e:
    print(f"Error: Module not found. {e}")
    sys.exit()

def test_meteor():
    print("Initializing Java Meteor...")
    try:
        # 1. Init
        meteor_scorer = Meteor()
        
        # 2. Data
        gts = {
            "0": ["con meo dang nam tren ghe", "meo nam ghe"],
            "1": ["troi hom nay rat dep"]
        }
        res = {
            "0": ["con meo nam tren ghe"],
            "1": ["thoi tiet hom nay dep"]
        }

        print("--- Computing metrics ---")
        
        # 3. Score
        avg_score, scores = meteor_scorer.compute_score(gts, res)

        print(f"Scores: {scores}")
        print(f"Average METEOR: {avg_score:.4f}")
        
        if avg_score > 0:
            print("\n- SUCCESS: Meteor is working!")
        else:
            print("\n- WARNING: Score is 0.")

    except Exception as e:
        print(f"\n- ERROR RUNNING METEOR:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_meteor()