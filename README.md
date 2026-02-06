## Streamlit App 
Check the deployed streamlit interface here: https://plantleaf-potatodiseaseclassifier.streamlit.app/ 

## Sample Example of Streamlit interface:
<img width="610" height="831" alt="image" src="https://github.com/user-attachments/assets/1d3faa0e-1a1b-492c-ba85-0237099a3221" />

<img width="647" height="526" alt="image" src="https://github.com/user-attachments/assets/5369e50a-f54f-4610-9ab8-fe67e9f78ec9" />



## Description
Here are two Component of the PlantLeaf project.
1. Potato Disease Classifier
2. Plant leaf disease classification model

Discussing the First model in Detail: 
1. Potato Disease Classifier:
   
  -> Here we have used Images of potato leaf classes from PlantVillage dataset.
  
  -> Further we have trained the **2152** images of potato leaf having **3 different classes** (see the figure below for Images distribution) by two ways
   ![image](https://github.com/user-attachments/assets/1b002cfc-4893-4d26-94eb-0ec357ab924a)

       -> Training the Custom CNN model from scratch
       -> Training on VGG16 model utilizing Transfer learning approach
   The test accuracy for Custom CNN model is more than that of VGG16 model.
  
  Test accuracy of Custom **CNN** Model = **98.44%**
  
  Test accuracy of **VGG16** model = **98.23%**

Discussing the second Model in Detail:

2. Plant leaf disease classification model:

   -> Here the entire PlantVillage dataset, having **50000+** Images, distributed into **38 different classes** containing images for 14 different plants species, is used for training.

   -> The VGG16 model is used for training the PlantVillage dataset, where the achieved test Accuracy is **93.75%.**
