import predictor
import sys


if predictor.model_pipe is None:
    print("Error: Model not loaded. Please run train_go_model.py")
    sys.exit(1)

print("🧠 GoEmotions Model Loaded. Type 'exit' to quit.")
print("-" * 50)

while True:
    try:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        prediction = predictor.predict_emotion(user_input)
        print(f"Detected Emotion: {prediction}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")