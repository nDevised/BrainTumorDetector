# Brain Tumor Detector

The Brain Tumor Detector is a deep learning project that aims to classify brain MRI images into two categories: "No Tumor" and "Tumor" using convolutional neural networks (CNN). This project demonstrates how to preprocess the MRI images, build a CNN model, train it, and use it for inference.

## Dataset

The dataset used in this project consists of brain MRI images, categorized into two classes:

1. No Tumor: MRI images of the brain without any tumors.
2. Tumor: MRI images of the brain with tumors.

The dataset should be organized as follows:

```
data/
    ├── no/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── yes/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Dependencies

Before running the Brain Tumor Detector, make sure you have the following dependencies installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Scikit-learn (`sklearn`)
- TensorFlow (`tensorflow`)
- Keras (`keras`)

You can install these dependencies using `pip`:

```
pip install opencv-python numpy scikit-learn tensorflow keras
```

## How to Use

1. Clone the repository to your local machine.
2. Organize your brain MRI dataset according to the specified structure.
3. Adjust the `imgPath` variable in the `main()` function to point to the directory containing the dataset.
4. Run the script `brain_tumor_detector.py` to train the model and save it.

The model will be trained on the dataset and saved as `BT10Ep.h5` in the project directory.

## Training the Model

The model architecture consists of several convolutional and pooling layers followed by fully connected layers for classification. The model is trained using binary cross-entropy loss and the Adam optimizer.

## Inference

Once the model is trained and saved, you can load it and use it for inference on new brain MRI images. The model will classify the input image as "No Tumor" or "Tumor" based on its features learned during training.

## Contributors

This project was developed by [Your Name] and [Contributor 2], and it serves as an educational example for deep learning beginners interested in medical image classification.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to acknowledge the dataset creators and the organizations that made the brain MRI data available for research and development purposes. Their contributions enable the advancement of medical imaging and artificial intelligence.
