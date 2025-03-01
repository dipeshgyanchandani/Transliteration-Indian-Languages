# **Transliteration and Image Captioning Models**

This repository contains the implementation of two key components aimed at solving the problem of **machine transliteration** and **image captioning**. The project focuses on **transliteration between English and Hindi** using a **sequence-to-sequence (Seq2Seq) model** and **image captioning** to detect and extract text from signboard images.

---

## **Key Components**

### **1. Image Captioning Model**

This model detects and extracts text from images of signboards. It uses an object detection framework, specifically **YOLO**, to identify text regions in images. The text is then passed to the image captioning model, which generates a readable string.

- **Object Detection (YOLO)**: Detects the text regions in the image.
- **Image Captioning Model**: Converts the detected text regions into a readable format (string).

- **Technology Stack**:
  - **YOLO** for text detection
  - **PyTorch** for training and deploying the model
  - **Python** for data preprocessing and model development

### **2. Transliteration Model**

The **Seq2Seq (Sequence-to-Sequence) Encoder-Decoder model** with **LSTM (Long Short-Term Memory)** architecture is used to perform transliteration between English and Hindi. The model is designed to handle long-range dependencies and improve the accuracy of transliteration using an **attention mechanism**.

- **Seq2Seq Model**: Encodes the input sequence and decodes it to the desired output sequence.
- **Attention Mechanism**: Enhances the model’s performance by focusing on specific parts of the input sequence when generating the output.
  
- **Technology Stack**:
  - **PyTorch** for building the model
  - **LSTM** for sequence processing
  - **Python** for data processing, training, and model evaluation

---

## **System Design**

The system design of the transliteration and image captioning pipeline is as follows:
<img width="882" alt="Screenshot 2025-02-14 at 1 39 23 AM" src="https://github.com/user-attachments/assets/452b66f6-60c9-45ed-8db2-bad5605e362f" />

### **Flow Overview:**

1. **Image Upload**: An image containing text (e.g., signboard image) is provided as input.
2. **Text Detection**:
   - The image is passed through the **object detection model (YOLO)** to detect text regions.
3. **Text Extraction**:
   - The detected regions of text are extracted and passed to the **image captioning model** for generating the readable text.
4. **Transliteration**:
   - The extracted text is sent to the **transliteration model (Seq2Seq LSTM)** for converting it from **English** to **Hindi** (or vice versa).
5. **Output**: The transliterated text is returned.


### **Technologies Used:**
- **YOLO** for object detection
- **Image Captioning Model** for text extraction
- **Seq2Seq with LSTM and Attention** for transliteration
- **PyTorch** for model development
- **Python** for data processing and running the models

---

## **Installation**

### **Prerequisites**

Ensure you have the following installed:

- Python 3.x
- PyTorch
- TensorFlow (if you plan to use pre-trained models)
- YOLO (for object detection, if you're using it)

### **Next Steps**

1. **Integrate the Models into a Web or Mobile Application**  
   You can integrate the object detection, image captioning, and transliteration models into a web or mobile application. For example:
   - Use **React Native** to develop a mobile application.
   - Set up a **Django backend** to handle the image upload, object detection, text extraction, and transliteration processes in real-time.

2. **Fine-Tune the Models with More Data**  
   To improve the accuracy of the models and handle more diverse input scenarios (e.g., various fonts, languages, or noisy images):
   - Fine-tune the object detection, image captioning, and transliteration models with a larger and more diverse dataset.
   - This will improve the model's generalization and performance across different signboards and languages.

3. **Extend the Transliteration Model to Support More Languages**  
   Expand the current transliteration model to support transliteration between additional language pairs, such as:
   - English ↔ Spanish
   - Hindi ↔ Marathi
   - Hindi ↔ Punjabi, etc.
   - You can also explore integrating this model with additional language datasets to support transliteration of multiple languages.

4. **Real-World Testing**  
   Once the models are integrated into an app or service, conduct real-world testing with users to:
   - Measure accuracy in different environments.
   - Collect user feedback for improvements in the app's UI/UX and model performance.

5. **Deploy the Application**  
   After successful testing, deploy the mobile or web application for real-time usage:
   - Set up cloud services to host the backend (e.g., AWS, Google Cloud, etc.).
   - Use mobile app deployment tools (e.g., App Store, Google Play) for wider user reach.

By implementing these steps, you can enhance the functionality of your transliteration system and bring it to practical use.

### **Clone the Repository**

```bash
git clone https://github.com/dipeshgyanchandani/Transliteration-Indian-Languages.git
cd Signboard-Translation

