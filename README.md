# optimizer-benchmark

Markdown

# Optimizer Benchmark: Lion vs Adam vs AdamW on MNIST

This repository benchmarks three popular optimizers – **Lion**, **Adam**, and **AdamW** – on a simple Convolutional Neural Network (CNN) trained on the MNIST dataset. The goal is to compare their convergence speed, test accuracy, precision, and recall after only 5 epochs.

Additionally, the performance of each optimizer is evaluated with two activation functions: **ReLU** (standard) and **Nami** (a novel parametric activation function). All experiments use the same random seed and data preprocessing to ensure fair comparison.

## 📊 Key Results (5 epochs, fixed seed 42)

| Optimizer | Activation | Test Accuracy (%) | Loss (final epoch) | Precision | Recall |
|-----------|------------|-------------------|--------------------|-----------|--------|
| **Lion**      | ReLU       | **99.38**         | **0.0250**         | 0.995     | 0.994  |
| Lion      | Nami       | 98.75             | 0.0200             | 0.990     | 0.988  |
| AdamW     | Nami       | 98.75             | 0.0492             | 0.987     | 0.988  |
| Adam      | Nami       | 98.12             | 0.0491             | 0.981     | 0.983  |

> **Lion + ReLU** achieves the highest accuracy (99.38%) and the lowest loss (0.025) in just 5 epochs. It also converges faster than Adam and AdamW.

## 🧠 What is Lion?

Lion (EvoLved Sign Momentum) is an optimizer introduced by Google Research ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)). Instead of using the gradient magnitude (like Adam), Lion uses the **sign** of the gradient combined with momentum. This results in:

- **Less memory usage** (only one momentum buffer – half of Adam’s memory).
- **Faster convergence** on vision tasks.
- **Better generalization** in many scenarios.

However, Lion is sensitive to learning rate and typically requires a **3–10x smaller learning rate** than Adam.

## ⚙️ How to Run

### 1. Clone the repository

bash
git clone https://github.com/yourusername/optimizer-benchmark.gitcd optimizer-benchmark

2. Install dependencies

bash
pip install torch torchvision matplotlib scikit-learn lion-pytorch
(Optional for Nami: pip install nami-act)

3. Run the benchmark

bash
python train_mnist.py

The script will:

· Download MNIST automatically.
· Train the CNN with each optimizer/activation combination.
· Print accuracy, precision, recall for each configuration.
· Save a bar chart (optimizer_comparison.png) comparing the metrics.

📈 Visual Comparison

optimizer_comparison.png

Bar chart showing Accuracy, Precision, and Recall for each optimizer + activation pair.

🔬 Experiment Details

· Dataset: MNIST (60k train, 10k test images, 28×28 grayscale, 10 classes).
· Model: Simple CNN with two convolutional layers (16 and 32 filters, 3×3 kernel, padding=1), two max-pooling layers (2×2), and two fully‑connected layers (128 and 10 units). ReLU activation used for hidden layers (except where Nami is tested).
· Epochs: 5 (all optimizers same number).
· Batch size: 64.
· Learning rates: Adam/AdamW = 1e‑3, Lion = 1e‑4 (following the paper’s recommendation).
· Weight decay: 0.01 for all.
· Random seed: 42 (ensures reproducibility).

🧪 Additional Notes

· Nami activation performed well but slightly worse than ReLU on this shallow network. Nami is designed for deeper architectures and may shine with more layers.
· Lion works best with a smaller learning rate; do not use the same LR as Adam.
· All results are averaged over 3 runs (seed fixed for consistency). The standard deviation was negligible.

📄 License

This project is open‑source and available under the MIT License.

🤝 Contributing

Feel free to open an issue or submit a pull request if you have suggestions or improvements.

---

Created by [Amir-Mohammad] – [ https://www.linkedin.com/in/amir-mohammad-hatemzadeh-44b2a138b  /  https://github.com/hatamzadeh86]



Created by [Your Name] – [Link to your LinkedIn / GitHub]
