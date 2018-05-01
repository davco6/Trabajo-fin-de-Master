### TFM
Trabajo fin de master.

Master en bioinformática y bioestadística de la UOC.

2018

Autor: David García Seisdedos

Estructura del repositorio:

  Data: contiene ejemplos de imágenes de microscopía y otras imágenes que serán usadas como ejemplo en los
        cuadernos de Jupyter.
				
  Creación de base de datos de imágenes con células: cuaderno en Jupyter que muestra como se realiza la obtención de los parches de los núcleos celulares segmentados. Así como, se muestra como se lleva a cabo el
        mezclado de dos imágenes nucleares para obtener una sola imagen con agregados nucleares.
				
  Segmentación individualizada de agrupaciones celulares: cuaderno en Jupyter que muestra el resultado de aplicar diferentes métodos de segmentación en imágenes con agregados nucleares.
				
  Segmentación de células en preparaciones microscópicas: cuaderno en Jupyter que muestra el resultado de aplicar diferentes métodos de segmentación de núcleos celulares en imágenes de microscopía.
				
  Synthetic_DB_v3.py: script en python por el que se obtienen imágenes sintéticas de núcleos celulares a partir
        de imágenes nucleares en bruto. 
				
  cell_training_vgg16.py: script en python para re-entrenar una red convolucional de arquitectura VGG16. Se analizan tres clases: imágenes con un solo núcleo, imágenes con agregados nucleares o imágenes con restos 
        celulares.
