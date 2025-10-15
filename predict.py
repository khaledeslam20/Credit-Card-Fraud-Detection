import argparse
import pandas as pd
from sklearn.metrics import f1_score
from credit_fraud_val_utils import load_model, save_classification_report, save_confusion_matrix


def predict_with_model(model_path, test_path, output_dir="predictions"):
    """
    Load a trained model and make predictions on test data
    """
    # Load the model
    model_dict = load_model(model_path)

    # Load test data
    test_df = pd.read_csv(test_path)
    x_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']

    # Apply scaling if scaler exists
    scaler = model_dict.get("scaler")
    if scaler is not None:
        x_test_scaled = scaler.transform(x_test)
    else:
        x_test_scaled = x_test

    # Make predictions
    model = model_dict["model"]
    threshold = model_dict.get("threshold", 0.5)
    y_probs = model.predict_proba(x_test_scaled)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Calculate and print metrics
    f1 = f1_score(y_test, y_pred)
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Model config: {model_dict.get('config')}")
    print(f"Threshold used: {threshold}")

    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save classification report
    save_classification_report(y_test, y_pred,
                               model_name=model_dict.get('model_name', 'unknown'),
                               experiment_name="test",
                               suffix="final",
                               output_dir=output_dir)

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred,
                          model_name=model_dict.get('model_name', 'unknown'),
                          experiment_name="test",
                          suffix="final",
                          save_dir=output_dir)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability': y_probs
    })
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"Predictions saved to {output_dir}/predictions.csv")

    return y_pred, y_probs, f1


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained credit fraud detection model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.pkl)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default="predictions",
                        help='Directory to save prediction results')

    args = parser.parse_args()

    # Make predictions
    y_pred, y_probs, f1 = predict_with_model(args.model_path, args.test_path, args.output_dir)

    print(f"\nPrediction completed!")
    print(f"F1 Score: {f1:.4f}")
    print(f"Results saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()