# Proyecto: 
# PEC2
# Synthetic_DB.py
# El pipeline crea nuevas imagenes a partir de las imágenes originales. Para ello,
# se lleva a cabo la rotación (rot), se cambia la prespectiva (shear),
# se da la vuelta (flip), se reescala la célula (zoom) y se añade ruido Gaussiano al azar.

####################
# Librerias
#####################
import os
import sys
import subprocess as sp
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import skimage
from skimage.measure import label, regionprops
from skimage import io
from skimage import transform as tf
from skimage.filters import threshold_li
##############



#############
## Funciones
#############
def rot_shear_zoom(image,num_images, lim_zoom):
    """
    Transforma la imagen inicial por una modificada mediante la libreria keras.preprocessing.image.
    Argumentos: 
        image: matriz de la imagen (numpy matrix),
        num_images: número de imagenes a generar (int),
        lim_zoom: límite por encima y por debajo del zoom a  aplicar (list, len=2)  
    """
    images_per_cell = (int(num_images)//len(images))
    datagen = ImageDataGenerator(
        rotation_range = 270,
        width_shift_range = 0.4,
        height_shift_range = 0.4,
        shear_range = 0.4,
        zoom_range = lim_zoom,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest')
    img = load_img(image)
    x = img_to_array(img)
    x= x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x,
                              batch_size = 1,
                              save_to_dir = "./synthetic/I",
                              save_prefix = "cat",
                              save_format = "tif"):
        i += 1
        if i > images_per_cell:
            break                       

def GaussianError(image):
    """
    Genera ruido gaussiano aleatorio.
    Argumentos:
        image: matriz de la imagen (numpy matrix).
    """
    #Gaussian Error
    error = np.random.choice(range(1, 21))
    brigth = np.random.choice(range(70, 200))
    for i, j in enumerate(list(np.random.choice(range(box), box*error))):
        if image[j, i%box] > 20:
            image[j, i%box] = np.random.normal(brigth, error, 1)
        if image[j, i%box] >= 255:
            image[j, i%box] = 255
    return (image)

def IC(cell, box):
    """
    Guarda la imágen y la modifica
    Argumentos:
        cell: la imagen a modificar (str),
        box: tamaño de imagen (int)
    """
    cell="./synthetic/I/"+cell
    print("Modifing cell: " + str(cell))
    image = io.imread(cell)
    image = image[:,:,0]
    image = GaussianError(image)
    img_warp=np.matrix(image, "uint8")
    im=Image.fromarray(img_warp)
    im.save(cell)
    return 

def rot(image, escalado=True):
    """
    Rota la imagen
    Argumentos: 
        image: matriz de la imagen (numpy matrix).
        escalado: decidir si se requiere escalado de la imagen (bool)
    """
    img = np.zeros((box, box))
    img[0:box, 0:box] = image
    if escalado == False:
        ei,ed = 1,1
    else:
        ei = random.uniform(0.9,1.1)
        ed = random.uniform(0.9,1.1) 
    rotation = random.uniform(-.5,.5)
    shear = random.uniform(-.5,.5)
    tform = tf.AffineTransform(scale = (ei,ed),
                          rotation = rotation,
                          translation = (0, 0),
                          shear = shear)
    img_warp = tf.warp(img, tform)
    
    if np.max(img_warp)<=10:
        return rot(image)
    else:        
        #Centrado de la imagen
        t_li = img_warp > 1
        label_image_blue = label(t_li)
        cell_region = regionprops(label_image_blue, img_warp)[0]
        cell_cen = cell_region.centroid
            
        tform = tf.AffineTransform(scale=(ei,ed),
                          rotation=rotation,
                          translation=(cell_cen[1]-box/2,cell_cen[0]-box/2),
                          shear=shear)
        img_warp = tf.warp(img, tform)
        if np.max(img_warp)==0:
            return rot(image)
        else:        
            return (img_warp)

def flip(cell):
    """
    Da la vuela a la imagen.
    Argumentos: 
        cell: imagen a modificar (str).
    """
    index = [np.random.choice(6),np.random.choice(2), np.random.choice(3)]
    if index[0] > 0:
        img=np.flip(np.rot90(cell, k=index[2]), index[1])
    else:
        img=np.rot90(cell, k=index[2])
    return cell

def MC(cellA, cellB, box):
    """
    Mezcla imágenes de células individuales para obtener imágenes con dos células
    Argumentos:
        cellA: primera célula a combinar (str)
        cellB: segunda célula a combinar (str)
        box: tamaño de la imagen en pixeles (int)
    """
    print("Mix: " + str(images[cellA])+" and "+ str(images[cellB]))
    A = io.imread("./I/"+images[cellA])
    A = flip(A)
    if cellA==cellB:
        B = rot(A, escalado=False)
    else:
        B = io.imread("./I/"+images[cellB])
    B = flip(B)
    A_exp = np.zeros((box*3,box*3))
    A_exp[box:box*2,box:box*2] = A
    #Cell A
    t_li = A_exp > 1
    label_image_blue = label(t_li)
    cell = regionprops(label_image_blue, A_exp)[0]
    A_contour = np.array(skimage.measure.find_contours(A_exp,0.5)[0], dtype="int")
    A_contour = np.unique(A_contour, axis=0).tolist()
    A_cen = cell.centroid
    random_lim = A_contour[np.random.choice(range(len(A_contour)))]
    #Cell B
    t_li = B > 1
    label_image_blue = label(t_li)
    cell = regionprops(label_image_blue, B)[0]
    B_rad = int(cell.equivalent_diameter*np.random.uniform(.8,1)/2)
    B_coords = cell.coords.tolist()
    B_cen_i = [int(cell.centroid[0]),int(cell.centroid[1])]
    # Combinado de las célula A y B
    angle = np.angle(np.complex((random_lim[1] - A_cen[1]), (random_lim[0] -A_cen[0])))
    B_cen = [int(random_lim[0] + (B_rad*np.sin(angle))),
           int(random_lim[1]+(B_rad*np.cos(angle)))]
    B_in_A = [B_cen[0] - B_cen_i[0], B_cen[1] - B_cen_i[1]]
    tmp = np.random.choice(range(3))
    if tmp > 0:
        for i in B_coords:
            A_exp[B_in_A[0] + i[0], B_in_A[1]+i[1]] = B[i[0], i[1]]
    else:
        for i in B_coords:
            A_exp[B_in_A[0] + i[0], B_in_A[1] + i[1]] = np.max([B[i[0], i[1]], A_exp[B_in_A[0] + i[0], B_in_A[1] + i[1]]])
    A = A_exp[box:box*2, box:box*2]
    if np.sum(A) < (box*box*.001*100):
        return (MC(np.random.choice(len(images)), np.random.choice(len(images)), box))
    img_warp = np.matrix(A, "uint8")
    
    im = Image.fromarray(img_warp)
    im.save("".join(["./II/",str(random.randint(1,999999)),".tif"]))#save
    plt.close("all")
    return 1

def Escalado(images):
    """
    Obtiene la desviación estandar del conjunto de células seleccionadas y calcular el mínimo y máximo zoom aplicable a cada célula, de tal manera que las imágenes generadas tengan un volumen comprendido entre las dos desviaciones estandard con respecto a la media.
    Argumentos: image: matriz de la imagen (numpy matrix).
    """
    vol = []
    for i in images:
        image = io.imread(i)
        t_li = image > 2
        label_image_blue = label(t_li)
        cell = regionprops(label_image_blue, image)[0]
        vol.append(cell.area)
    ds = [(np.mean(vol)-np.std(vol))/np.mean(vol), (np.mean(vol)+np.std(vol))/np.mean(vol)]
    return ds

def eliminate_white_cell(list_cells,lim):
    """
    Elimina las imagenes que no contienen ninguna célula o imagenes con células pegadas a los bordes.
    Argumentos:
        list_cells: lista de imágenes de células. List(str)
        lim: número limite de imagenes para generar, int.
    """
    for i in list_cells:
        image = io.imread("./synthetic/I/"+i)
        image = image[:,:,0]
        int_border = sum(image[0,:])+sum(image[-1,:])+sum(image[:,0])+sum(image[:,-1])
        if np.max(image) <= 50 or int_border > 0:
            sp.call([" ".join(["rm", "./synthetic/I/"+i])], shell=True)
    synt_unicell = [s for s in os.listdir("./synthetic/I") if ".tif" in s]
    if len(synt_unicell) >= lim:
        return
    else:
        rot_shear_zoom(np.random.choice(images),lim-len(synt_unicell), Zoom)
        synt_unicell = [s for s in os.listdir("./synthetic/I") if ".tif" in s]
        return eliminate_white_cell(synt_unicell,lim)

def mix_cell_position(image, box):
    """
    Cambia de posición el núcleo celular en la imagen.
    Argumentos:
        image: matriz de la imagen (numpy matrix).
        box: tamaño de imagen (int).
    """
    img = io.imread(image)
    t_li = img > 1
    label_image_blue = label(t_li)
    cell = regionprops(label_image_blue, img)[0]
    cell_cen = [int(cell.centroid[0]),int(cell.centroid[1])]
    cast = np.zeros((box*3,box*3))
    random_cen = [np.random.choice(range(box)),np.random.choice(range(box))]
    cast[box-cell_cen[0]:2*box-cell_cen[0] ,box-cell_cen[1]:2*box-cell_cen[1]]= img
    sel = cast[3//2*box-random_cen[0]:3//2*box-random_cen[0]+box,3//2*box-random_cen[1]:3//2*box-random_cen[1]+box]
    col_sel_cel = np.matrix(sel, "uint8")
    im = Image.fromarray(col_sel_cel)
    im.save(image) 
#######
# Fin de funciones
##########

wd = sys.argv[1]
box = int(sys.argv[2])
lim = int(sys.argv[3])

os.chdir(wd)

if __name__ == "__main__":
    images = [s for s in os.listdir(".") if ".tif" in s]
    Zoom = Escalado(images) #Obtenemos los límites de escalado según el tamaño de las células seleccionadas
    sp.call([" ".join(["mkdir","synthetic"])], shell=True)
    sp.call([" ".join(["mkdir","synthetic/I"])], shell=True)
    sp.call([" ".join(["mkdir","synthetic/II"])], shell=True)

    # Modificamos las imagenes de células originales para obtener lim imagenes modificadas
    for image in images:
        rot_shear_zoom(image, lim, Zoom)

    synt_unicell = [s for s in os.listdir("./synthetic/I") if ".tif" in s]
    eliminate_white_cell(synt_unicell, lim)#Eliminamos imágenes que no contienen ninguna célula
    cells = [s for s in os.listdir("./synthetic/I") if ".tif" in s]

    #Generamos ruido gaussiano en las imagenes generadas y las guardamos
    for cell in cells: 
        IC(cell, int(box))
    os.chdir("./synthetic")
    images = [s for s in os.listdir("./I") if ".tif" in s]
    i = 0
    while i < int(lim):
        #Mezclamos imagenes de núcleos para obtener imagenes con dos núcleos.
        cell_A = np.random.choice(len(images))
        cell_B = np.random.choice(len(images))
        if i%5 != 0: 
            i += MC(cell_A, cell_B, int(box))
        else:
            i += MC(cell_A, cell_A, int(box))

    #Cambia de posición los nucleos de las imagenes generadas (sólo se realiza en las imagenes con un solo nucleo).
    os.chdir("./I")
    images = [s for s in os.listdir(".") if ".tif" in s]
    for image in images:
        mix_cell_position(image, box)


