# Entry point for HCMC AI Challenge 2026 inference pipeline
import argparse
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print("AI Challenge 2026 Pipeline Initialized successfully!")
    print(f"Loaded config: {config}")

if __name__ == "__main__":
    main()
