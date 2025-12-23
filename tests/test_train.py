from src.train import main

def test_training_runs():
    main(test_size=0.2, random_state=42)
