# Ultralytics Utility Scripts Collection

This repository contains a collection of useful standalone scripts designed to support workflows built with **Ultralytics YOLO**.
The scripts focus on practical tasks commonly needed in real-world projects such as dataset preparation, model training, evaluation, and explainability.

Most scripts are intentionally lightweight and can be copied directly into your own project with minimal modification.

---

# Available Features

## Dataset Preparation and Management

This repository contains useful scripts related to dataset preparation and management:

* [YOLO Dataset Splitter](#yolo-dataset-splitter)
* [LabelMe to YOLO Conversion](#labelme-to-yolo-conversion)
* [Visualization of YOLO Annotations](#visualize-yolo-annotations)

---

## Model Evaluation and Metrics

This repository contains useful scripts related to calculation and evaluation for Ultralytics models:

* [YOLO K-Fold Evaluation](#yolo-k-fold-evaluation)

---

## Model Training

This repository contains useful scripts related to model training:

* [YOLO K-Fold Training](#yolo-k-fold-training)

---

## Model Explainability

This repository contains useful scripts related to model interpretability and explainability:

* [Multi-Layer EigenCAM](#multi-layer-eigencam)

---

# Usage

These scripts are designed to be simple and reusable.

In most cases, you can use them by:

1. Copying the script into your project
2. Adjusting paths and parameters
3. Running the script

```bash
python script_name.py
```

No installation as a package is required.

---

# Development Status

⚠️ **Important Notice**

* Not all scripts in this repository have been fully tested in every environment
* Some scripts may still contain bugs or edge-case issues
* APIs and structure may change in future updates

Use the scripts as utilities and adapt them as needed for your specific project.

---

# Future Improvements

This repository is actively evolving. Planned improvements include:

* Adding more utility scripts
* Refactoring duplicated code
* Creating a proper modular project structure
* Improving documentation and examples
* Standardizing configuration and CLI interfaces
* Integrating shared utilities to reduce redundancy

Currently, most scripts are **independent** and not tightly integrated with each other.
This design keeps them flexible but may result in some code duplication.

A more structured architecture may be introduced in future versions.

---

# Philosophy

This repository prioritizes:

* Practical utility
* Reusability
* Simplicity
* Engineering workflow support

The goal is not to build a framework, but to provide reliable building blocks for real projects.

---

# License

You are free to use, modify, and adapt these scripts for your own projects.
