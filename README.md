# Automated Object Classification for CCTV Recording

## 📋 Problem Description

Surveillance is a critical aspect of modern industries, ensuring safety and security. However, traditional CCTV systems record continuously, leading to excessive memory consumption and inefficient storage use. With advancements in artificial intelligence, it is now possible to automate CCTV recording based on object detection and classification.

This project aims to develop an intelligent system that activates CCTV recording **only** when specific objects (such as humans, vehicles, or animals) are detected in the frame. By doing so, unnecessary recordings are avoided, significantly reducing memory requirements and improving surveillance efficiency.

## 🚀 Project Objective

- **Build a deep learning model** capable of classifying objects in CCTV images into predefined categories (e.g., human, vehicle, animal, etc.).
- **Integrate the model** into a CCTV pipeline to trigger recording only when relevant objects are detected.
- **Evaluate** the model using a benchmark dataset (Fashion-MNIST) before deploying with real-world CCTV data.

## 📊 Dataset

For initial development and testing, the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is used:

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image size:** 28x28 pixels, grayscale
- **Number of classes:** 10

Each image is labeled as one of ten clothing categories, making it suitable for benchmarking image classification models.

## 🏗️ Solution Approach

1. **Data Preprocessing**
   - Normalize pixel values to [0,
   - Reshape images for model compatibility
   - One-hot encode labels (if needed)

2. **Model Architecture**
   - Use a Convolutional Neural Network (CNN) for image classification
   - Layers include convolution, pooling, dropout, and dense layers

3. **Training & Validation**
   - Train the model on the Fashion-MNIST training set
   - Validate using a subset of the training data
   - Fine-tune hyperparameters (learning rate, batch size, epochs, etc.)

4. **Evaluation**
   - Test the model on the Fashion-MNIST test set
   - Analyze accuracy, loss, and confusion matrix
   - Visualize training and validation performance

5. **Deployment (Future Scope)**
   - Adapt the model for real CCTV footage using datasets such as COCO or ImageNet
   - Integrate with CCTV systems to automate recording based on object detection

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn (for analysis and visualization)
- Jupyter Notebook (for prototyping)

## 📈 Results

- The CNN model achieved a **test accuracy of 92.1%** on the Fashion-MNIST dataset.
- Training and validation accuracy curves showed good convergence, with minimal overfitting.
- Confusion matrix analysis indicated strong class-wise performance, with most misclassifications occurring between visually similar classes.
- These results demonstrate the feasibility of using deep learning for automated, object-based CCTV recording[5][3].

## 📦 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cctv-object-classification.git
   cd cctv-object-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook or script**
   ```bash
   # For Jupyter Notebook
   jupyter notebook cctv_object_classification.ipynb

   # For Python script
   python cctv_object_classification.py
   ```

## 📚 References

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

## 📄 License

This project is licensed under the MIT License.

---

**Project maintained by Gowtham B.**  
For questions or suggestions, please contact gowthammourya9@gmail.com.

[1] https://sist.sathyabama.ac.in/sist_naac/aqar_2022_2023/documents/1.3.4/b.e-cse-batchno-134.pdf
[2] https://eocortex.com/object-classification-and-counting
[3] https://cs229.stanford.edu/proj2017/final-reports/5234577.pdf
[4] https://bprd.nic.in/uploads/pdf/12%20CCTV.pdf
[5] https://ece.anits.edu.in/2019-20%20BE%20Project%20REPORTS/CHPS%20_2019-2020%20project%20batch_2.pdf
[6] https://sist.sathyabama.ac.in/sist_naac/documents/1.3.4/b.tech-it-batchno-29.pdf
[7] https://www.scribd.com/document/427923761/Object-Detection-Project-Report-docx
