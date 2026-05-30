# soil-moisture-pinn
PINN model on soil data

# Soil Moisture Prediction Using Physics-Informed Neural Networks (PINN)

## About the Project

This project uses a Physics-Informed Neural Network (PINN) to predict soil moisture values from sensor data.

The goal is to combine machine learning with simple physical constraints so that the model learns both from data and from physics-based relationships.

This project was developed as part of my learning in Data Science and Machine Learning.

---

## Dataset

The dataset contains the following columns:

* time
* moisture0 (target)
* moisture1
* moisture2
* moisture3
* moisture4

The model uses:

Input Features:

* time
* moisture1
* moisture2
* moisture3
* moisture4

Target:

* moisture0

---

## Project Structure

```text
data/
models/
train/
utils/
visualize/
main.py
```

### Folders

* data/ : dataset files
* models/ : PINN model architecture
* train/ : training scripts
* utils/ : preprocessing and physics functions
* visualize/ : result visualization scripts

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

---

## Model

The project uses a Physics-Informed Neural Network (PINN).

The loss function contains:

1. Data Loss (MSE Loss)
2. Physics Loss

The total loss is:

Total Loss = Data Loss + Physics Loss

---

## Training

To train the model:

```bash
python train/train_pinn.py
```

---

## Visualization

To visualize predictions:

```bash
python visualize/plot_results.py
```

The visualization shows:

* Real vs Predicted Moisture
* Prediction Error
* Error Distribution

---

## Learning Outcomes

Through this project I learned:

* Data preprocessing
* Neural Networks with PyTorch
* Physics-Informed Neural Networks (PINNs)
* Training and validation workflows
* Model visualization
* Git and GitHub

---

## Future Improvements

Possible future improvements:

* Better physics equations
* More sensor data
* Hyperparameter tuning
* Advanced PINN architectures
* Deployment as a web application

---

## Author

Ashish Ranjan Tiwari

Data Science Student
