**Ensemble Stack for Face Presentation Attack Detection**
*July 2024 – November 2024*

---

## 1. Project Overview

Biometric authentication systems increasingly rely on face recognition for secure access. However, these systems are vulnerable to presentation attacks—spoofing attempts using printed photos, replayed videos, or 3D masks—that can compromise security. Detecting these spoofing threats in real time, particularly under diverse lighting and environmental conditions, remains a key challenge in computer vision research.

**Objective:**
Develop a hybrid ensemble model that combines global and local facial feature analysis to robustly detect presentation attacks with high accuracy and low latency, suitable for deployment on edge devices.

## 2. Technical Approach

### Dual-Model Architecture

* **Vision Transformer (ViT):** Leverages self-attention to capture spatial and structural inconsistencies indicative of spoofing (e.g., unnatural edges, lack of micro-movements).
* **EfficientNet-B7:** Focuses on fine-grained texture patterns—such as print artifacts or screen moiré—to augment global analysis with detailed surface cues.

### Transfer Learning & Training

1. **PyTorch Transfer Learning:**

   * Initialized ViT and EfficientNet-B7 from pre-trained ImageNet weights.
   * **Layer Freezing Strategy:** Locked lower convolutional and transformer blocks to retain generic feature representations; fine-tuned higher layers for spoof-specific patterns.
2. **Mixed-Precision Training:**

   * Employed NVIDIA Apex for FP16 training to accelerate convergence and reduce memory footprint.
   * Trained on NVIDIA A100 GPUs with batch sizes optimized for throughput.

### Deployment Pipeline

1. **TensorRT Optimization:** Converted PyTorch models to ONNX and applied TensorRT quantization and kernel auto-tuning to achieve sub-50 ms inference latency on edge hardware.
2. **OpenCV Integration:**

   * Face detection and alignment using Haar cascades or DNN-based detectors.
   * Standardized preprocessing (resizing, color normalization) for consistent model input.

## 3. Datasets & Evaluation

* **Primary Datasets:**

  * **CASIA-FASD** and **Replay-Attack**: Benchmarked model performance on diverse attack mediums (print, video, mask).
* **Cross-Dataset Generalization:**

  * Tested on **OULU-NPU** and **MSU-MFSD** to evaluate robustness against domain shifts.
* **Interpretability Techniques:**

  * Generated attention maps overlaying facial regions that contributed most to spoof detection.
  * Visualized blink irregularities and texture anomalies to validate model focus areas.

## 4. Results

* **Ensemble Performance:**

  * **Accuracy:** 97% overall.
  * **Individual Models:** ViT – 93%; EfficientNet-B7 – 94%.
* **Error Reduction:**

  * 15% reduction in Equal Error Rate (EER) compared to single-model baselines.
* **Efficiency Gains:**

  * **Training Throughput:** Improved by 40% using mixed-precision.
  * **Inference Speed:** < 50 ms per frame on edge GPUs post-TensorRT.
* **Domain Shift Resistance:** Demonstrated a 12% improvement in cross-dataset accuracy when evaluated on OULU-NPU and MSU-MFSD.

## 5. Installation & Usage Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/username/ensemble-stack-fasd.git
   cd ensemble-stack-fasd
   ```
2. **Environment Setup:**

   ```bash
   conda create -n fasd_env python=3.9
   conda activate fasd_env
   pip install -r requirements.txt
   ```

   **Dependencies:** PyTorch ≥1.13, NVIDIA Apex, ONNX, TensorRT, OpenCV, matplotlib, scikit-learn.
3. **Running Inference:**

   ```bash
   python inference.py --model ensemble.onnx --input image.jpg --output result.jpg
   ```
4. **Custom Testing:**

   * Replace `image.jpg` with your own inputs or a folder of test images.
   * Use `--batch_size` and `--device` flags to adjust performance settings.

## 6. Visualizations & Examples

* **Attention Map Samples:** Highlight regions—such as eye corners and facial edges—exposed to spoofing artifacts.
* **Confusion Matrix & ROC Curves:** Included in `/docs/visuals` illustrating true positive vs. false positive rates.
* **Performance Charts:** Training loss curves and EER trends across datasets.

## 7. Future Work

* **Temporal Modeling:** Integrate LSTM or 3D-CNN layers to exploit sequential frame cues (e.g., natural eye blinks).
* **Adversarial Robustness:** Apply adversarial training to defend against gradient-based spoofing attacks.
* **Large-Scale Deployment:** Extend support for mobile and ARM-based platforms; automate continuous model updates via federated learning.

---

**Contributors:**

* Raahul Muthukumar (Lead Researcher and Developer)


**License:**
This project is licensed under the MIT License.
