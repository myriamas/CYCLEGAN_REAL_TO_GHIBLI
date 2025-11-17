# CycleGAN for Real ↔ Ghibli Image Translation

Here's the link to my notebook with all the outputs and the visualizations of the results : https://colab.research.google.com/drive/1PkKxciuNz02ytoGnloDP9ZB0Ri85bMGq#scrollTo=SBFPMzZw77PB


## Overview
This project implements a full **CycleGAN** architecture to translate images between two domains:
- **Real-world images → Ghibli-style images**
- **Ghibli-style images → Real-world images**

The dataset contains **2500 real images** and **2500 Ghibli-style artistic images**, unpaired.  
Our objectives were:
1. Generate new Ghibli-style images from unseen real photos.  
2. Reconstruct a realistic image from a stylized Ghibli image.

CycleGAN was chosen because it is specifically suited for **unpaired image-to-image translation**.

---

## Dataset
- Source: Kaggle — *Real to Ghibli Dataset* (5000 high-quality images)  
- Two independent folders:  
  - `trainA`: real images  
  - `trainB_ghibli`: Ghibli-style images  
- Images vary in resolution, so a preprocessing pipeline was built to standardize them.

---

## Preprocessing Pipeline
Implemented in TensorFlow for clean integration with the GAN:
- Load and resize images to **256×256**
- Normalize pixels to **[-1, 1]**
- Convert to tensors
- Optional augmentation (random flip)
- Batch, shuffle, prefetch
- Zip the two domains to prepare CycleGAN training

This produces efficient `tf.data.Dataset` objects for training.

---

## Model Architecture

### **Generators (Real→Ghibli and Ghibli→Real)**
- U-Net-style encoder–decoder  
- Skip connections for spatial preservation  
- LayerNormalization for training stability  
- Final activation: **tanh** (matches normalized pixel range)

### **Discriminators**
- PatchGAN architecture  
- Operates on local patches instead of full images  
- Encourages texture-level realism

### **Loss Functions**
- Adversarial loss (BCE)
- Cycle-consistency loss  
- Identity loss  
- **Perceptual loss (VGG19)** to improve visual realism

These losses ensure that:
- Real→Ghibli→Real reconstructs the original
- Ghibli→Real→Ghibli reconstructs the original
- Outputs stay within the target domain distribution

---

## Training Strategy
Training was done in **three stages**:

### **1. Pretraining Real→Ghibli (G_RG / D_G)**  
Warm-up to stabilize stylization quality.

### **2. Pretraining Ghibli→Real (G_GR / D_R)**  
Warm-up for the inverse mapping.

### **3. Full CycleGAN Joint Training**  
- Both generators and discriminators trained together  
- Cycle, identity, adversarial, and perceptual losses combined  
- Models saved regularly for progress tracking  
- Intermediate visualizations produced every few epochs  

---

## Results

### **Real → Ghibli**
- The model generates visually coherent stylized images  
- Successfully captures color palette, brush texture, and artistic patterns  
- Outputs are stable and aesthetically consistent

### **Ghibli → Real**
- Much more challenging  
- Outputs lack sharp realism  
- Stylized inputs do not contain enough information to reconstruct real-world detail  
- This task is inherently asymmetric: stylization = simplification; destylization = reconstruction of missing details

---

## Project Structure
- Data loading and preprocessing  
- GAN component definitions  
- Pretraining loops  
- Full CycleGAN training loop  
- Visualization utilities  
- Demo output generation  
- Model saving / checkpoints  

---

## Technologies
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- VGG19 (ImageNet features for perceptual loss)  
- Kaggle dataset integration

---

## Authors
Project developed by Myriam Ait Said.

