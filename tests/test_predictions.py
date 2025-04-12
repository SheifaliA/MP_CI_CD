import sys
from pathlib import Path
# Dynamically determine file paths for resolving module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to sys.path for module imports
sys.path.append(str(root))
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from vehicleinsurance_model.predict import make_prediction



def test_make_prediction(sample_input_data):
    """Test function to validate the model's prediction process."""

    # Given: Expected number of predictions based on dataset size
    expected_num_of_predictions = 76222

    # When: Execute the prediction function
    result = make_prediction(input_data=sample_input_data[0])

    # Then: Validate the prediction output
    predictions = result.get("predictions")
    print(type(predictions[0]))  # Print type for debugging

    # Ensure predictions are returned as a NumPy array of integer values
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array."
    assert isinstance(predictions[0], np.int64), "Each prediction should be of type np.int64."
    assert result.get("errors") is None, "Expected no errors during prediction."
    assert len(predictions) == expected_num_of_predictions, f"Expected {expected_num_of_predictions} predictions, but got {len(predictions)}."

    # Convert predictions to a list for metric evaluation
    _predictions = list(predictions)
    y_true = sample_input_data[1]

    # Calculate accuracy and precision metrics
    ACC = accuracy_score(y_true, _predictions)
    PRE = precision_score(y_true, _predictions)

    # Validate model performance thresholds
    assert ACC > 0.8, f"Model accuracy too low: {ACC}. Expected > 0.8."
    assert PRE > 0.3, f"Model precision too low: {PRE}. Expected > 0.3."