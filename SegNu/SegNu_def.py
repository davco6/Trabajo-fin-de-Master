# David Garcia Seisdedos
# SegNu
# Definitions
# May 2018

#####################################
# Load libraries
#####################################
import sys
import os
import subprocess as sp
import numpy as np


from skimage import io, color
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.transform import resize

from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as k

from scipy import ndimage as ndi

from PIL import Image
import time
####################################
# Definitions
####################################

def apply_watershed (binary_image, min_distance, raw_image):
    """
    Apply watershed segmentation.

    :param binary_image: A numpy matrix. Binary image.
    :param min_distance: A float value. Minimum numberof pixels separating local maximum peaks
    :param raw_image: A numpy matrix. Original image.
    """
    distance = ndi.distance_transform_edt (binary_image)
    local_maxi = peak_local_max (distance,
                                indices = False, 
                                labels = binary_image,
                                min_distance = min_distance)
    markers = ndi.label(local_maxi)[0]
    label_image = watershed(-distance, markers = markers, mask = binary_image)
    label_image *= binary_image
    regions = regionprops(label_image, raw_image)
    return (regions, label_image)

def generate_patches(regions, box, label_image, original_image, CELL_NUM=0, original_patch_size=192):
    """
    Generate patches with the input regions.

    :param regions: A list of scikit-image regions.
    :param box: An integer value. Number of the pixels in the side of the patch.
    :param label_image: A numpy matrix. Labeled image.
    :param original_image: A numpy matrix. Original image.
    :param CELL_NUM: A integer value. Cell counter.
    :param original_patch_size: A numpy matrix. Original image.
    """
    input_shape = original_image.shape
    region_image = label_image.copy()
    for region in regions:
        region_image[:,:] = 0
        region_image[label_image == region.label] = region.label
        region_image[region_image != 0] = original_image[region_image != 0]
        centre = region.centroid
        bbox = [(int(centre[0]-box/2)), (int(centre[0]+box/2)), (int(centre[1]-box/2)), (int(centre[1]+box/2))]
        for i,j in enumerate(bbox):
            if j < 0:
                bbox[i+1] = box
                bbox[i] = 0
            if j > input_shape[0] and i == 1:
                bbox[i-1] = input_shape[0] - box
                bbox[i] = input_shape[0]
            if j > input_shape[1] and i == 3:
                bbox[i-1] = input_shape[1] - box
                bbox[i] = input_shape[1]
                          
        CELL_NUM += 1
        sel_cel = region_image[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        sel_cel = resize(sel_cel, (original_patch_size,original_patch_size), mode="symmetric", preserve_range=True)
        sel_cel = sel_cel*255/sel_cel.max()
        col_sel_cel = np.matrix(sel_cel, "uint8")
        img = Image.fromarray(col_sel_cel)
        img.save("".join(["/tmp/SegNu/Samples/",str(CELL_NUM+1000000),".tif"]))
        
    return 

def CNN_inference(dim_patch, num_regions):
    """
    Generate patches with the input regions.

    :param dim_patch: An integer value. Number of the pixels in the side of the patch.
    :param num_regions: An integer value. Number of the region to be analyse.
    """
    nb_classes = 3
    img_width, img_height = dim_patch,dim_patch
    batch_size = 16
    n = 16
    samples_data_dir = '/tmp/SegNu'
    model_pretrained = './weights/model_weights.h5'
    base_model = VGG16(input_shape = (dim_patch, dim_patch, 3), weights = 'imagenet', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    model.load_weights(model_pretrained)
    predict_data = ImageDataGenerator(rescale=1. / 255)
    validation_generator = predict_data.flow_from_directory(samples_data_dir,
                                                            target_size = (img_width, img_height),
                                                            batch_size = batch_size,
                                                            class_mode = None,
                                                            shuffle = False)
    predictions = model.predict_generator(validation_generator)
    predictions = predictions.argmax(axis=1).tolist()
    k.clear_session()
    return (predictions)

def set_arg(argv):
    """
    Set the input and output arguments

    :param argv: Input arguments.
    """
    import getopt
    outputpath = ""
    segmentation = "watershed"
    save_images = False
    try:
        opts, args = getopt.getopt(argv,"hi:o:t:s",["ipath=","opath=", "segmentation=", "save_images"])
    except getopt.GetoptError:
        print ("ERROR")
        print ('SegNu.py -i <inputpath> -o <outputpath> -t <type_of_segmentation> -s <save_images>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('\nSegNu.py -i <inputpath> -o <outputpath> -s <save_images> -t <type_of_segmentation>\n\n<inputpath> path to the working images directory\n<outputpath> path to the output file\n<type_of_segmentation> choose a valid type of segmentation: clustering or watershed\n<save_images>wether or not save the processed images')
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputpath = arg
        elif opt in ("-o", "--opath"):
            outputpath = arg
        elif opt in ("-s", "--save_images"):
            save_images = True
        elif opt in ("-t", "--type_of_segmentation"):
            if arg not in ["clustering_AC", "watershed", "only_limit_threshold","watershed_no_cnn", "clustering_gauss"]:
                print("Error")
                print("Type of segmentation argument fail, choose between: clustering, watershed or empty (watershed segmentation by defect)")
            else:
                segmentation = arg
    if outputpath == "":
        outputpath = inputpath + "/Results"
    print ("Input path is " + inputpath)
    print ("Output path is " + outputpath)
    print ("Type of segmentation: " + segmentation)
    print ("Save images? " + str(save_images))
    return (inputpath, outputpath, segmentation, save_images)

def create_folders(output_folder):
     """
     Create the working folders.
     """
     sp.call([" ".join(["mkdir","/tmp/SegNu"])],shell=True)
     sp.call([" ".join(["mkdir","/tmp/SegNu/Samples"])],shell=True)
     sp.call([" ".join(["mkdir",output_folder])],shell=True)
     sp.call([" ".join(["mkdir",output_folder+"/Instances_images"])],shell=True)
     sp.call([" ".join(["mkdir",output_folder+"/Modified_images"])],shell=True)

def AC (binary_image, original_image):
    """
    Apply agglomerative clustering segmentation.

    :param binary_image: A numpy matrix. Binary image.
    :param original_image: A numpy matrix. Original image.
    """
    from sklearn.cluster import AgglomerativeClustering
    lcc = label(binary_image)
    area = [i.area for i in regionprops(lcc,original_image)]
    region = regionprops(lcc,original_image)[area.index(max(area))]
    AC = AgglomerativeClustering(n_clusters = 2, linkage = "average").fit(region.coords)
    cast = np.zeros((original_image.shape[0],original_image.shape[1]))
    c = AC.fit_predict(region.coords)
    for i in range(region.coords.shape[0]):
        cast[region.coords[i,0],region.coords[i,1]]=c[i]+1
    cast = np.matrix(cast, "uint8")
    regions = regionprops(cast, original_image)
    return (regions, cast)

def Gaussian (binary_image, original_image):
    """
    Apply Gaussian clustering mixture segmentation.

    :param binary_image: A numpy matrix. Binary image.
    :param original_image: A numpy matrix. Original image.
    """
    from sklearn import mixture
    lcc = label(binary_image)
    area = [i.area for i in regionprops(lcc,original_image)]
    region = regionprops(lcc,original_image)[area.index(max(area))]
    Gauss = mixture.GaussianMixture(n_components = 2,
                                covariance_type = 'full').fit(region.coords)

    cast = np.zeros((original_image.shape[0],original_image.shape[1]))
    c = Gauss.predict(region.coords)
    for i in range(region.coords.shape[0]):
        cast[region.coords[i,0],region.coords[i,1]]=c[i]+1
    cast = np.matrix(cast, "uint8")
    regions = regionprops(cast, original_image)
    return (regions,cast)

def rm_tmp_files():
     """
     Remove temporary files.
     """
     sp.call([" ".join(["rm","/tmp/SegNu/Samples/*"])], shell=True)

def run_watershed_seg(threshold_steps,
                      radio,
                      unicell_regions,
                      cluster_regions,
                      label_image,
                      raw_image,
                      patch):
    """
    Run watershed segmentation recursively.

    :param threshold_steps: A list of floats.
    :param radio: An integer value. Expected cell radio. 
    :param unicell_regions: A list of unicell scikit-image regions.
    :param cluster_regions: A list of cells clusters scikit-image regions.
    :param label_image: A numpy matrix. Labeled image.
    :param raw_image: A numpy matrix. Original image.
    :param patch: An integer value. Leght (in pixels) of the side of the sqare patch.
    """
    from skimage.morphology import closing, square
    if len(threshold_steps) == 0 or cluster_regions == []:
        segmented_image = np.zeros((label_image.shape[0], label_image.shape[1]))
        unicell_regions += cluster_regions
        for h,i in enumerate(unicell_regions):
            i.label = h+1
            for j in range(i.coords.shape[0]):
                segmented_image[i.coords[j,0],i.coords[j,1]] = i.label
        return (segmented_image, unicell_regions)
    else:
        region_image = np.zeros((label_image.shape[0],label_image.shape[1]))
        image_leftovers = region_image.copy()
        for h,j in enumerate(cluster_regions):
            region_image[label_image == j.label] = j.label
            image_leftovers[region_image != 0] = raw_image[region_image != 0]
            region_image[:,:] = 0
        binary_image = closing(image_leftovers > threshold_steps[0], square(3))
        regions_splitted, label_image = apply_watershed(binary_image, radio,raw_image) 
        generate_patches (regions_splitted, patch, label_image,raw_image)
        predictions = CNN_inference(192,len(regions_splitted))
        unicell_regions += [j for i, j in enumerate(regions_splitted) if predictions[i] == 1]
        cluster_regions = [j for i,j in enumerate(regions_splitted) if predictions[i] == 2]
        rm_tmp_files()
        return (run_watershed_seg(threshold_steps[1:], radio, unicell_regions, cluster_regions, label_image, raw_image, patch))

def run_clustering_seg(unicell_regions,
                       cluster_regions,
                       label_image,
                       raw_image,
                       patch,
                       seg):
    """
    Run agglomerative clustering segmentation recursively.
 
    :param unicell_regions: A list of unicell scikit-image regions.
    :param cluster_regions: A list of cells clusters scikit-image regions.
    :param label_image: A numpy matrix. Labeled image.
    :param raw_image: A numpy matrix. Original image.
    :param patch: An integer value. Length (in pixels) of the side of the sqare patch.
    """
    seg=seg
    if cluster_regions == []:
        segmented_image = np.zeros((label_image.shape[0],label_image.shape[1]))
        for h,i in enumerate(unicell_regions):
            i.label = h+1
            for j in range(i.coords.shape[0]):
                segmented_image[i.coords[j,0],i.coords[j,1]] = i.label
        return (segmented_image, unicell_regions)
    else:
        region_image = np.zeros((label_image.shape[0],label_image.shape[1]))
        clt_regions = []
        for n,m in enumerate(cluster_regions):
            if cluster_regions[n].area > 9:
                region_image[label_image == cluster_regions[n].label] = cluster_regions[n].label
                if seg=="clustering_AC":
                    regions_splitted,label = AC(region_image, raw_image)
                if seg=="clustering_gauss":
                    regions_splitted,label = Gaussian(region_image, raw_image)
                generate_patches(regions_splitted,patch, label,raw_image,CELL_NUM=n*2)
                clt_regions += regions_splitted
                region_image[:,:] = 0

        predictions = CNN_inference(192,len(clt_regions))
        unicell_regions += [j for i, j in enumerate(clt_regions) if predictions[i] == 1]
        cluster_regions = [j for i,j in enumerate(clt_regions) if predictions[i] == 2]
        label_image[:,:] = 0
        rm_tmp_files()
        for h,i in enumerate(cluster_regions):
            i.label = h+1
            for j in range(i.coords.shape[0]):
                label_image[i.coords[j,0],i.coords[j,1]] = i.label
        return (run_clustering_seg(unicell_regions, cluster_regions, label_image, raw_image, patch, seg))




def initial_segmentation(image,
                         PATCH_RATIO):
    """
    Initialization of the image segmentation.
 
    :param image: A numpy matrix. Original image.
    :param PATCH_RATIO: Constant. Ratio between patch area against cell mean cell area.
    """
    from skimage.filters import threshold_li, threshold_otsu, threshold_minimum
    from skimage.morphology import closing, square
    thresh_li = threshold_li(image)
    thresh_otsu = threshold_otsu(image)
    try:
        thresh_min = threshold_minimum(image)
    except RuntimeError:
        thresh_min = thresh_otsu + 100

    if thresh_min < thresh_otsu:
        threshold_steps = range(int(thresh_li ), int(thresh_otsu), abs(int((thresh_otsu-thresh_li)/5)))
    else:
        threshold_steps = [thresh_otsu]
    binary_image = closing(image > threshold_steps[0], square(3))
    label_image = label(binary_image)
    regions = regionprops(label_image, image)
    regions_above_noise = []
    areas = []
    for region in regions:
        if region.area >= 9:
            #Lista con las nuevas regiones
            areas.append(region.area)
            regions_above_noise.append(region)

    median=np.median(areas)
    patch = int((PATCH_RATIO*median)**(0.5))
    return (threshold_steps,patch, regions_above_noise, label_image)

def save_mod_images(raw_image,segmented_image, unicell_regions,output_dir, file):
    """
    Saving the segmented image over the orignal image
 
    :param raw_image: A numpy matrix. Original image.
    :param segmented_image: A numpy matrix. Labeled image.
	:param unicell_regions: Constant. A list of unicell scikit-image regions.
	:param output_dir: String. Output directory.
	:param file: String. Input image path.
    """
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    fig, ax = plt.subplots(figsize=(10,10))
    CELL_NUM = 1
    label_image=segmented_image.copy()
    for region in unicell_regions:
        segmented_image[:,:] = 0
        segmented_image[label_image == region.label] = region.label
        contour=find_contours(segmented_image,0.5)
        count = [i.shape for i in contour]
        contour=contour[count.index(max(count))]
        ax.plot(contour[:,1], contour[:,0],'r', linewidth=1)
        minr, minc, maxr, maxc = region.bbox
        ax.text(0.5*(minc+maxc),0.5*(minr+maxr), str(CELL_NUM), horizontalalignment="center", verticalalignment="center", fontsize=10, color="blue")
        CELL_NUM += 1
    ax.imshow(raw_image, cmap='gray')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("".join([output_dir,"/Modified_images/",file.split(".")[0],".tif"]), dpi=fig.dpi,bbox_inches="tight")
    return 

def save_instances_image(segmented_image, output_dir,file):
    """
    Saving the segmented and labeled image
 
    :param segmented_image: A numpy matrix. Labeled image.
	:param output_dir: String. Output directory.
	:param file: String. Input image path.
    """
    segmented_image=np.matrix(segmented_image, "uint8")
    img=Image.fromarray(segmented_image)
    img.save("".join([output_dir,"/Instances_images/",file.split(".")[0],".tif"]))
    

def patch_tune(images_list, PATCH_RATIO, tmp_regions=[]):
    """
    Tune the patch length
 
    :param images_list: A numpy matrix. Labeled image.
	:param PATCH_RATIO: Constant. Ratio between patch area against cell mean cell area.
    """
    if len(tmp_regions)>100 or images_list==[]:
        areas = [region.area for region in tmp_regions]
        mean = np.mean(areas)
        patch = int((PATCH_RATIO*mean)**(0.5))
        radio = int((mean/np.pi)**0.5)
        return (patch, radio)
    else:
        image = io.imread(images_list[0])
        if len(image.shape) == 3:
            image = image[:,:,2]                
        _, patch,regions_above_noise, label_image = initial_segmentation(image,PATCH_RATIO)
        generate_patches(regions_above_noise, patch,label_image,image)
        
        predictions = CNN_inference(192,len(regions_above_noise))
        rm_tmp_files()
        regions_without_debris = [j for i,j in enumerate(regions_above_noise) if  predictions[i]>0]
        tmp_regions += regions_without_debris
        return (patch_tune(images_list[1:], PATCH_RATIO, tmp_regions))


def run(PATCH_RATIO):
    wd, od, segmentation, save_images = set_arg(sys.argv[1:])
    create_folders(od)
    #rm_tmp_files()
    extensions = ['.jpg', '.jpeg', '.tif', '.TIF', '.JPG', '.JPEG']
    images_list = []
    with open(od+"/Results.txt", "w") as f:
        f.write("\t".join(["Image", "Number of cell nuclei"])+"\n")
    print("Loading files...\n")
    for file in os.listdir(wd):
        for extension in extensions:
            if file.endswith(extension):   
                images_list.append(os.path.join(wd, file))
    
    print("Tuning zoom patch...\n")        
    patch,radio = patch_tune(images_list, PATCH_RATIO, tmp_regions=[])
    print("Set new dimention patch:"+str(patch))
    print("End tuning zoom patch.\n")
    t0 = time.time()
    for path_image in images_list:
        print ("Analyzing image: "+path_image.split("/")[-1])
        image = io.imread(path_image)
        if len(image.shape)==3:
            image = image[:,:,2]
        threshold_steps,patch,regions_above_noise, label_image = initial_segmentation(image,PATCH_RATIO)
        if segmentation=="only_limit_threshold" or segmentation=="watershed_no_cnn":
            if segmentation=="only_limit_threshold":
                unicell_regions, segmented_image = regions_above_noise, label_image
            if segmentation=="watershed_no_cnn":
                regions, label_image= apply_watershed (label_image, radio, image)
                unicell_regions, segmented_image = regions, label_image
        else:
            generate_patches(regions_above_noise, patch,label_image,image)
            predictions = CNN_inference(192,len(regions_above_noise))
            regions_without_debris = [j for i,j in enumerate(regions_above_noise) if  predictions[i]>0]
            predictions=[i for i in predictions if i!=0]
            cluster_regions = [j for i,j in enumerate(regions_without_debris) if  predictions[i]==2 ]
            unicell_regions = [i for i in regions_without_debris if i not in cluster_regions]
            rm_tmp_files()
        
            if segmentation=="watershed":
                segmented_image, unicell_regions = run_watershed_seg(threshold_steps,radio,unicell_regions,cluster_regions,label_image, image, patch)
            else:
                segmented_image, unicell_regions = run_clustering_seg(unicell_regions,cluster_regions,label_image, image, patch, segmentation)
        
        save_instances_image(segmented_image, od, path_image.split("/")[-1])
        with open(od+"/Results.txt", "a") as g:
            g.write("\t".join([path_image.split("/")[-1], str(len(unicell_regions))])+"\n")
        print("Number of cells found: "+str(len(unicell_regions)))
        if save_images:
            save_mod_images(image,segmented_image, unicell_regions,od, path_image.split("/")[-1])
            
    t1 = time.time()
    print((t1-t0)/float(len(images_list)))
