# 🧠 Micrograd: A Tiny Autograd Engine

A from-scratch implementation of automatic differentiation (like PyTorch).

## 📌 Overview

This project implements **Micrograd**, a minimal autograd engine inspired by Andrej Karpathy’s [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLqnslRFeYoVZDRUcJjk8e6mttwyHj0N5f) series. It includes:

* A `Value` class to track gradients via dynamic computation graphs.
* Backpropagation for scalar-valued functions.
* A toy neural net trained with gradient descent.

**Key Achievement:** Built PyTorch-like autograd with <200 lines of Python.

## 🚀 Features

✅ **Automatic Differentiation**

* Supports `+`, `*`, `tanh`, `exp`, and `ReLU` operations.
* Computes gradients via recursive backpropagation.

✅ **Neural Network Training**

* Demo: 2-layer MLP trained on toy data.
* Extensible to custom architectures.

✅ **Visualized Computation Graphs**

* Example:

    ```python
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b  # Dynamically builds graph
    ```

    (Generated via `draw_dot`)

    ![computation_graph_example](computation_graph_example.png)

## 📦 Installation

```bash
git clone [https://github.com/Pranshulx26/karpathy-nn-zero-to-hero](https://www.google.com/search?q=https://github.com/Pranshulx26/karpathy-nn-zero-to-hero)
cd karpathy-nn-zero-to-hero/1_micrograd

jupyter notebook Micrograd.ipynb


