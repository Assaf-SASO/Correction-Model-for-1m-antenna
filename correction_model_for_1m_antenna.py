import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Create a new output folder with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'output_{timestamp}')
os.makedirs(output_dir)
print(f"Created new output directory: {output_dir}")

# Frequency data (MHz)
frequencies = np.array([500, 600, 700, 800, 900, 1000, 4000, 5000, 6000, 7000, 8000])
delta_cf = np.array([-3.86, -3.01, -3.27, -2.76, -2.71, -2.97, -2.97, -3.07, -3.22, -3.16, -3.32])

# CF data at 1m
freq_1m = np.array([500, 600, 700, 800, 900, 1000, 4000, 5000, 6000, 7000, 8000])
cf_1m = np.array([-3.35, -3.10, -2.85, -2.50, -2.97, -2.38, -2.97, -3.61, -3.48, -3.88, -5.04])

# Reshape for sklearn
X = frequencies.reshape(-1, 1)
y = delta_cf

# Try different polynomial degrees to find the best fit
degrees = [2, 3, 4, 5]
models = []
mse_scores = []
r2_scores = []

plt.figure(figsize=(12, 10))

# Plot original data
plt.scatter(frequencies, delta_cf, color='blue', label='Observed Data')

# Create a denser frequency range for smoother curve plotting
X_plot = np.linspace(min(frequencies), max(frequencies), 1000).reshape(-1, 1)

for degree in degrees:
    # Create polynomial regression model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_plot = model.predict(X_plot)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Store results
    models.append(model)
    mse_scores.append(mse)
    r2_scores.append(r2)
    
    # Plot the polynomial fit
    plt.plot(X_plot, y_plot, label=f'Degree {degree} (MSE: {mse:.4f}, R²: {r2:.4f})')

# Find the best model based on MSE
best_idx = np.argmin(mse_scores)
best_model = models[best_idx]
best_degree = degrees[best_idx]

print(f"Best polynomial degree: {best_degree}")
print(f"Mean Squared Error: {mse_scores[best_idx]:.4f}")
print(f"R² Score: {r2_scores[best_idx]:.4f}")

# Function to predict Delta CF for any frequency
def predict_delta_cf(frequency):
    """
    Predict the Delta CF value for a given frequency using the trained model.
    
    Args:
        frequency (float): Frequency in MHz
        
    Returns:
        float: Predicted Delta CF value in dB
    """
    return best_model.predict(np.array([[frequency]]))[0]

# Example prediction for a new frequency
test_freq = 2500  # MHz
predicted_delta = predict_delta_cf(test_freq)
print(f"Predicted Delta CF for {test_freq} MHz: {predicted_delta:.4f} dB")

# We don't need a separate model for CF at 1.0 m as we'll use the actual measured values
# and apply the delta CF correction to them

# Function to apply correction to a 1.0 m measurement using the delta CF model
def apply_correction(frequency, measured_cf_1m):
    """
    Apply correction to a 1.0 m measurement based on the delta CF polynomial model.
    
    Args:
        frequency (float): Frequency in MHz
        measured_cf_1m (float): Measured CF value at 1.0 m in dB
        
    Returns:
        tuple: (corrected_cf, delta_cf)
            - corrected_cf (float): Corrected CF value in dB
            - delta_cf (float): The predicted delta CF value used for correction
    """
    # Get the predicted delta CF for this frequency
    predicted_delta_cf = predict_delta_cf(frequency)
    
    # Apply the correction by subtracting the delta CF from the measured value
    corrected_cf = measured_cf_1m - predicted_delta_cf
    
    return corrected_cf, predicted_delta_cf

plt.xlabel('Frequency (MHz)')
plt.ylabel('Delta CF (dB)')
plt.title('Delta CF vs Frequency with Polynomial Regression Models')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'frequency_correction_model.png'), dpi=300, bbox_inches='tight')

reference_freqs = np.linspace(500, 8000, 16)

example_freqs = [600, 1000, 5000, 8000]
example_measurements = [-3.20, -2.45, -3.70, -5.10]

def save_data_to_files():
    with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
        f.write('Polynomial Degree,MSE,R² Score\n')
        for degree, mse, r2 in zip(degrees, mse_scores, r2_scores):
            f.write(f'{degree},{mse},{r2}\n')
        f.write(f'\nBest polynomial degree: {best_degree}\n')
        f.write(f'Mean Squared Error: {mse_scores[best_idx]:.4f}\n')
        f.write(f'R² Score: {r2_scores[best_idx]:.4f}\n')
    
    print(f"\nModel Results (saved to {os.path.join(output_dir, 'model_results.txt')})")
    print(f"Best polynomial degree: {best_degree}")
    print(f"Mean Squared Error: {mse_scores[best_idx]:.4f}")
    print(f"R² Score: {r2_scores[best_idx]:.4f}")
    
    with open(os.path.join(output_dir, 'reference_predictions.txt'), 'w') as f:
        f.write('Frequency (MHz),Predicted Delta CF (dB)\n')
        for freq in reference_freqs:
            pred = predict_delta_cf(freq)
            f.write(f'{freq:.0f},{pred:.4f}\n')
    
    print(f"\nReference Predictions (saved to {os.path.join(output_dir, 'reference_predictions.txt')})")
    print("Frequency (MHz) | Predicted Delta CF (dB)")
    print("-" * 35)
    for freq in reference_freqs[::3]:  # Print every third value for brevity
        pred = predict_delta_cf(freq)
        print(f"{freq:14.0f} | {pred:15.4f}")
    
    with open(os.path.join(output_dir, 'corrected_measurements.txt'), 'w') as f:
        f.write('Frequency (MHz),Original CF (dB),Predicted Delta CF (dB),Corrected CF (dB)\n')
        for freq, orig_cf in zip(freq_1m, cf_1m):
            corrected, delta = apply_correction(freq, orig_cf)
            f.write(f'{freq:.0f},{orig_cf:.4f},{delta:.4f},{corrected:.4f}\n')
    
    print(f"\nCorrected Measurements (saved to {os.path.join(output_dir, 'corrected_measurements.txt')})")
    print("Frequency (MHz) | Original CF (dB) | Predicted Delta CF (dB) | Corrected CF (dB)")
    print("-" * 75)
    for freq, orig_cf in zip(freq_1m, cf_1m):
        corrected, delta = apply_correction(freq, orig_cf)
        print(f"{freq:14.0f} | {orig_cf:15.4f} | {delta:17.4f} | {corrected:15.4f}")
    
    print(f"\nRequired output files have been saved in: {output_dir}")
    print("Files generated:")
    print("- frequency_correction_model.png")
    print("- corrected_measurements.txt")
    print("- model_results.txt")
    print("- reference_predictions.txt")

save_data_to_files()

