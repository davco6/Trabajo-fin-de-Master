# TFM
## Trabajo fin de master. David García Seisdedos.

### Master en bioinformática y bioestadística de la UOC.

### 2018

### Director del trabajo fin de master:
### Edwin Santiago Alférez Baquero

### Estructura del repositorio:

* Data: contiene ejemplos de imágenes de microscopía y otras imágenes que serán usados como ejemplos en los
        cuadernos de Jupyter.
				
* Creación_de_la_base_de_datos: cuaderno en Jupyter que muestra como se realiza la obtención de los parches de los núcleos celulares segmentados. Así como, se muestra como se lleva a cabo el
        mezclado de dos imágenes nucleares para obtener una sola imagen con agregados nucleares.
				
* Segmentación_de_agrupaciones_nucleares: cuaderno en Jupyter que muestra el resultado de aplicar diferentes métodos de segmentación en imágenes con agregados nucleares.
				
* Segmentacion_de_nucleos: cuaderno en Jupyter que muestra el resultado de aplicar diferentes métodos de segmentación de núcleos celulares en imágenes de microscopía.
				
* Synthetic_DB_v3.py: script en python por el que se obtienen imágenes sintéticas de núcleos celulares a partir
        de imágenes nucleares en bruto. 
				
* cell_training_vgg16.py: script en python para re-entrenar una red convolucional de arquitectura VGG16. Se analizan tres clases: imágenes con un solo núcleo, imágenes con agregados nucleares o imágenes con restos 
        celulares.
	
* SegCNN-v3.ipynb: cuaderno en Jupyter que muestra el proceso de segmentación y clasificación de núcleos celulares que realiza el software SegNu.py.

* SegNu: Software escrito en python que segmenta núcleos celulares en imágenes de microscopía con un solo canal o tres canales (siempre que los núcleos estén marcados en azul). El software utiliza redes neuronales convolucionales para ayudar en la segmentación de agrupaciones nucleares. La segmentación de agregados nucleares puede realizarse por tres métodos: método de watershed, método de agrupación aglomerativa o método mixto gaussiano. Para que el software pueda funcionar es neceario la utilización del fichero de pesos, pero debido a su tamaño (aprox. 100 MBytes) no es posible subirlo al repositorio de GitHub.

### Construido con:

* [Python] (https://www.python.org/) - Lenguaje de programación utilizado.
* [Jupyter] (http://jupyter.org/) - Software de programción interactiva.
* [Keras] (https://keras.io/) - Conjunto de funciones y procedimientos para la implementación de redes neuronales.

### Autor:
* **David García Seisdedos**
