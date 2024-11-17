import os
from models.train import train_model
from models.predict import predict_answer

if __name__ == "__main__":
    # Train the model
    train_model()

    # Test predictions
    question_1 = "What is the capital of France?"
    context_1 = "France, located in Western Europe, has Paris as its capital and largest city."

    question_2 = "Who wrote the novel '1984'?"
    context_2 = "'1984' is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell in 1949."

    print(f"Answer 1: {predict_answer(question_1, context_1)}")
    print(f"Answer 2: {predict_answer(question_2, context_2)}")
