from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"

if not MODEL_PATH.exists():
    raise SystemExit("Model file missing. Run train.py first.")

model = joblib.load(MODEL_PATH)

examples = [
    (
        "spam",
        "Your parcel is on hold. Pay the 1.49 delivery fee at "
        "parcel-release.example within 2 hours to avoid its return.",
    ),
    (
        "spam",
        "Your mobile number won a brand new gaming laptop! "
        "Send a 20 euro processing fee to claim your reward today.",
    ),
    (
        "spam",
        "Earn 400 euros daily by liking videos from home. "
        "No interview needed. Pay your activation fee now to start!",
    ),
    (
        "spam",
        "URGENT: Your account will be locked in 30 minutes. "
        "Reply with your password and verification code to keep access.",
    ),
    (
        "ham",
        "Hey, I finished my Python project. Want to meet at the "
        "library tomorrow and try it out?",
    ),
    (
        "ham",
        "Could you pick up some bread on your way home? "
        "I already bought the milk.",
    ),
    (
        "ham",
        "Our machine learning assignment is due on Friday. "
        "Remember to add the results to your README.",
    ),
    (
        "ham",
        "I booked a table for four at 7 tonight. "
        "Let me know if you will be late.",
    ),
    (
        "ham",
        "I have a spare ticket for tonight's concert. You can have "
        "it for free if you want to come with us.",
    ),
    (
        "ham",
        "Our team won the school coding competition! "
        "We can collect our prize from the teacher tomorrow.",
    ),
    (
        "ham",
        "The urgent account warning you received is fake. "
        "Don't reply or share your password.",
    ),
]

print("=" * 70)
print("1. MESSAGES TO CHECK")
print("=" * 70)

for number, (expected, message) in enumerate(examples, start=1):
    print(f"\nMessage {number}")
    print(message)

messages = [message for expected, message in examples]
predictions = model.predict(messages)

label_names = {
    "ham": "Not spam",
    "spam": "Spam",
}

correct_count = 0
mistakes = []

print("\n" + "=" * 70)
print("2. PREDICTION RESULTS")
print("=" * 70)

print(
    f"{'Message':<10}"
    f"{'Expected':<15}"
    f"{'Predicted':<15}"
    f"{'Result'}"
)
print("-" * 55)

for number, ((expected, message), predicted) in enumerate(
    zip(examples, predictions),
    start=1,
):
    if predicted == expected:
        correct_count += 1
        result = "Correct"
    else:
        result = "Incorrect"
        mistakes.append((number, message, expected, predicted))

    print(
        f"{number:<10}"
        f"{label_names[expected]:<15}"
        f"{label_names[predicted]:<15}"
        f"{result}"
    )

print(f"\nCorrect predictions: {correct_count}/{len(examples)}")

print("\n" + "=" * 70)
print("3. INCORRECT PREDICTIONS")
print("=" * 70)

if not mistakes:
    print("\nNo incorrect predictions.")
else:
    for number, message, expected, predicted in mistakes:
        print(f"\nMessage {number}")
        print(message)
        print(f"Expected:  {label_names[expected]}")
        print(f"Predicted: {label_names[predicted]}")
        print("-" * 70)