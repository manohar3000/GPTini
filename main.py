"""
Main entry point for the language model toolkit.

This script provides a simple interface to either train or generate text.
All configurations are in config.py.
"""

def main():
    """Main entry point - ask user whether to train or generate."""
    print("PyTorch Language Model")
    print("1. Train model")
    print("2. Generate text")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nTraining model with parameters from config.py")
        from train import train
        train()
    elif choice == "2":
        model_path = input("Enter path to model checkpoint: ").strip()
        prompt = input("Enter text prompt (or press Enter for empty prompt): ").strip()
        try:
            length = int(input("Enter length to generate (default: 100): ").strip() or "100")
        except ValueError:
            length = 100
            
        from generate import generate
        generate(model_path, prompt, length)
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()