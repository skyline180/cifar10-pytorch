# CIFAR-10 Image Classification with PyTorch

## 📌 Overview
This project implements an image classification pipeline on the **CIFAR-10 dataset** using PyTorch.  
It follows a clean, modular structure with reusable components for data loading, model design, training, and evaluation.

---

## 📊 Dataset
- **CIFAR-10**
- 60,000 RGB images (32×32)
- 10 classes:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🧠 Model
- Custom Convolutional Neural Network (CNN)
- Architecture:
  - Conv → ReLU → Pool
  - Conv → ReLU → Pool
  - Fully Connected Layers

---

## ⚙️ Project Structure
```
cifar10-pytorch/
├── notebooks/ # experiments & visualization
├── src/ # core ML code
├── configs/ # configuration files
├── outputs/ # models, logs, figures
├── data/ # dataset (ignored in git)
└── main.py # training entry point
```


---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/cifar10-pytorch.git
cd cifar10-pytorch
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python main.py
```
---

## 📓 Notebooks

- `01_exploration.ipynb` → Dataset visualization  
- `02_training.ipynb` → Model training  
- `03_evaluation.ipynb` → Predictions & results  

## 📈 Results

| Metric   | Value |
|----------|--------|
| Accuracy | ~70%   |


## 🖼 Sample Predictions

<img width="950" height="433" alt="image" src="https://github.com/user-attachments/assets/bd4a3c0e-f66e-412d-8cd5-376e6b2aec88" />


# 🧑‍💻 Author

Md. Shahat Akash

## 📜 License

MIT License
