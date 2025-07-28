from src.preprocessing import load_data, preprocess
from src.model_training import train_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load
    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    # Preprocess
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Split features/labels
    X = train_df.drop(['Triage'], axis=1)
    y = train_df['Triage']

    X_test = test_df.drop(['Triage'], axis=1)
    y_test = test_df['Triage']

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    scores = evaluate_model(model, X_val, y_val)
    print("Validation Scores:", scores)

    # Final test evaluation
    test_scores = evaluate_model(model, X_test, y_test)
    print("Test Scores:", test_scores)

    # Save model
    save_model(model)
