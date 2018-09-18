# TFM
## Final work. David García Seisdedos.

### Bioinformatics and biostadistics Master Degree at UOC.

### 2018

### Director: Edwin Santiago Alférez Baquero

### Repository structure:

* Data: contains examples of microscopy images and other images that will be used as examples in the Jupyter notebooks
				
* Creación_de_la_base_de_datos: notebook in Jupyter that shows how to obtain the patches of the segmented cell nuclei. As well as, it shows how the mixing of two nuclear images is carried out to obtain a single image with nuclear aggregates.
				
* Segmentación_de_agrupaciones_nucleares: notebook in Jupyter that shows the result of applying different segmentation methods in images with nuclear aggregates.
				
* Segmentacion_de_nucleos: notebook in Jupyter that shows different nuclei segmentation methods in microscopy images.
				
* Synthetic_DB_v3.py: script in Python which generate synthetic nuclei aggregates from single nuclei. 
				
* cell_training_vgg16.py: script in Python which retrain a convolutional neural network with VGG-16 architecture. It is analyzed three classes of images with: single nuclei, nuclear aggregates and non nuclear elements.
	
* SegCNN-v3.ipynb: notebook in Jupyter that shows the segmentation and classification process done by software SegNu.py.

* SegNu: Software write in Python which apply a cell nuclei segmentation in microscopy images. SegNu uses CNN for the cell classification. The segmentation of nuclear aggregates could be done by: watershed method or by clustering method (agglomerative or Gaussian mixture).

### Programming tools used:

* [Python] (https://www.python.org/) 
* [Jupyter] (http://jupyter.org/) 
* [Keras] (https://keras.io/) 

### Author:
* **David García Seisdedos**
