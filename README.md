# Final Project — PatchCamelyon Classifier and Explainability

This repository contains a full pipeline to train a classifier on the [PatchCamelyon](https://patchcamelyon.grand-challenge.org/) dataset, analyze the results, and generate visual explanations using modern XAI (eXplainable AI) methods.

---

## Project Structure

- **checkpoints/**  
  Stores model checkpoints during training (e.g., `best_model.pth`).  

- **comparative_outputs/**  
  Contains Grad-CAM, Integrated Gradients, and LIME visualizations side-by-side for different samples.

- **models/**  
  Placeholder for model definitions or additional model files.

- **data_manipulation.py**  
  Functions to load, preprocess, and manage PatchCamelyon `.h5` files into PyTorch datasets.

- **EDA.ipynb**  
  Exploratory Data Analysis: dataset exploration, distribution of patches, visualization of samples.

- **explicability_pipeline.py**  
  Pipeline to run explainability methods (Grad-CAM, Integrated Gradients, LIME) and compute evaluation metrics like Deletion AUC and Insertion AUC.

- **model.py**  
  Training loop, checkpointing, early stopping, and ResNet-18 fine-tuning logic.

- **requirements.txt**  
  List of required Python packages to reproduce the environment.

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Final-Project-Rex.git
cd Final-Project-Rex
```

2.	Install dependencies

It is recommended to create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

3.	Download the dataset

Ensure the .h5 files for training, validation, and testing are available in a data/ directory.
(See EDA.ipynb for links and data exploration.)

4.	Train or load a model

You can either train a new model from scratch or load a checkpoint if available.

5.	Generate explanations

Run the explicability pipeline to create visualizations under comparative_outputs/.

---

## Explanation Methods Included
- Grad-CAM
Highlights regions most responsible for the model’s decision, using activations from the last convolutional layer.
- Integrated Gradients
Attributes importance by averaging gradients along a path from a baseline (black image) to the actual input.
- LIME
Locally approximates the model with a simple interpretable model by perturbing input regions (superpixels).

We also compute Deletion AUC and Insertion AUC metrics to quantitatively evaluate each method.

---

## Results Summary

| Method                 | Deletion AUC ↓ (lower = better) | Insertion AUC ↑ (higher = better) |
|-------------------------|---------------------------------|-----------------------------------|
| **Grad-CAM**            | 0.39                            | 0.80                             |
| **Integrated Gradients**| 0.62                            | 0.64                             |
| **LIME**                | 0.34                            | 0.72                             |


Interpretations and panel-by-panel analyses can be found inside the Analysing Results section of the project notebook or report.

---

## Notes
- The final model uses ResNet-18, fine-tuned by retraining only the last fully connected layer, to speed up training and reduce overfitting risks.
- Only 10% of the available training data was used, with stratification to ensure class balance.
- Data Augmentation was applied (horizontal flip, rotation, color jitter).

---

## Acknowledgements
- PatchCamelyon Dataset
- Captum library for Integrated Gradients
- LIME library
- PyTorch Grad-CAM

---

License

This project is for academic and educational purposes only.

---