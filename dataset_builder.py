import cv2
import os
import numpy as np   
import random
import tqdm
import math 
from PIL import Image

jpg_background_folder = 'C:\\Users\\artem\\Desktop\\work\\puncher\\cards_names\\background_jpg'
png_background_folder = 'C:\\Users\\artem\\Desktop\\work\\puncher\\cards_names\\background_png'
jpg_image_folder = 'C:\\Users\\artem\\Desktop\\work\\puncher\\cards_names\\all_cards_w_names_jpg'
png_image_folder = 'C:\\Users\\artem\\Desktop\\work\\puncher\\cards_names\\all_cards_w_names_png'

# adding empty alpha channel
def add_alpha_channel(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    return img

def create_pngs(srs_path, dest_path):
    for filename in os.listdir(srs_path):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            source_path = os.path.join(srs_path, filename)
            destination_path = os.path.join(dest_path, os.path.splitext(filename)[0] + ".png")
            # add an alpha channel to the image
            image_with_alpha = add_alpha_channel(source_path)
            # save the image with the alpha channel as PNG to the destination folder
            image_with_alpha.save(destination_path, format="PNG")

# !!!
# Change to False, if you already have PNGs with alpha channel
to_PNGs = False
if to_PNGs:
    create_pngs(jpg_background_folder, png_background_folder)
    create_pngs(jpg_image_folder, png_image_folder)

bg_list = []  
class_names = []
class_images = []

# load backgrounds
for f in os.listdir(png_background_folder):
    filepath = os.path.join(png_background_folder, f)
    image = cv2.imread(filepath)
    if (type(image) is np.ndarray): 
        bg_list.append(image)

# load class images
for f in os.listdir(png_image_folder):
    filepath = os.path.join(png_image_folder, f)
    classname = f[:-4]
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if (type(image) is np.ndarray):
        class_names.append(classname)
        class_images.append(image)


bg_amount = len(bg_list)
class_amount = len(class_names)


# overlay foreground image to background with alpha channels
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# place image to background in random location
def random_overlay_transparent(background, image, minprop: float, maxprop: float):
    '''
    Randomly place image on background
        background - background image
        image - overlay image
        minprop and maxprop - min and max relative image size to background
    '''
    # clip proportions to range [0.0, 1.0]
    minprop = np.clip(minprop, 0.0, 1.0)
    maxprop = np.clip(maxprop, 0.0, 1.0)

    # define random proportion in range
    prop = np.random.uniform(minprop, maxprop)
    
    # resize image
    bg_height, bg_width = background.shape[:2]    
    img_height, img_width = image.shape[:2]
    new_width = int(bg_width * prop)
    new_height =  int(new_width/img_width * img_height)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
 
    # define random placement location 
    x_pos, y_pos = random.randrange(0, bg_width-new_width), random.randrange(0, bg_height-new_height)
    
    # generate output
    # xywh
    bbox = (int(x_pos + new_width/2), int(y_pos + new_height/2), new_width, new_height)
    res_img = overlay_transparent(background, image_resized, x_pos, y_pos)
    return bbox, res_img

# rotate image with no corners cut
def rotate_image(img, angle):

    # make image squared
    h, w = img.shape[:2]
    vert, horiz = 0, 0
    if h>w: horiz = int((h-w)/2)  # if vertical image - add L+R borders
    if w>h: vert = int((w-h)/2)  # if horizontal image - add T+B borders

    # extend image for rotation
    H = int(img.shape[0]/(np.sqrt(2)/2) - img.shape[0]) # cos 45
    vert += H
    horiz += H

    image = cv2.copyMakeBorder(img, vert, vert, horiz, horiz, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    
    # dividing height and width by 2 to get the center of the image
    height, width = image.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    R = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=R, dsize=(width, height))

    return rotated_image

# remove transparent borders (alpha channel)
def cut_transparent_borders(image):
    '''
    Detects transparent borders in alpha channel and cuts image
    '''

    # find transparent borders
    alpha = image[:,:,3]
    x = np.flatnonzero(np.max(alpha, axis=0))[[0,-1]]
    y = np.flatnonzero(np.max(alpha, axis=1))[[0,-1]]

    return image[y[0]:y[1],x[0]:x[1],:]

# change overall brightness of image
def change_brightness(img, value):
    alpha = img[:,:,3]
    alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
    hsv = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return np.concatenate((img, alpha), axis=2)

def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    alpha = None
    if img.shape[2]>3:
        alpha = img[:,:,3]
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
    bgr = img[:,:,:3]
    brightness += int(round(255*(1-contrast)/2))
    bgr = cv2.addWeighted(bgr, contrast, bgr, 0, brightness)
    
    if alpha is not None:
        bgr = np.concatenate((bgr, alpha), axis=2)

    return bgr

# change value of each colorchannel separately
def change_color(img, delta_rgb):
    a = None
    if img.shape[2]>3:
        b, g, r, a = cv2.split(img)
    else:
        b, g, r = cv2.split(img)

    r = cv2.add(r,delta_rgb[0]) ;  r[r > 255] = 255; r[r < 0] = 0
    g = cv2.add(g,delta_rgb[1]) ;  g[g > 255] = 255; g[g < 0] = 0
    b = cv2.add(b,delta_rgb[2]) ;  b[b > 255] = 255; b[b < 0] = 0

    if a is not None:
        result = cv2.merge((b, g, r, a))
    else:
        result = cv2.merge((b, g, r))
    return result

# randomly place transparent blocks on image
def random_transparent_decay(image, maxrate):
    '''
    Place transparent blocks randomly on image

    rate - percentage of overall image area
    '''
    '''  
    rate = np.clip(rate, 0.0, 1.0)
    h, w = image.shape[:2]
    area = h * w # amount of pixels
    decay_area = int(area * rate) # approx amount of pixels to make transparent
    blocks = int(rate * 100) # amount of square blocks
    block_size = int(np.sqrt(decay_area/blocks)) # side of square block
    
    # place blocks
    for i in range(blocks):
        x_pos, y_pos = random.randrange(0, w-block_size), random.randrange(0, h-block_size)
        image[y_pos:y_pos+block_size, x_pos:x_pos+block_size, 3] = 0
    '''
    rate = np.random.uniform(0.0, maxrate)
    alpha =  image[:,:,3]
    image[:,:,3] = alpha * (1.0-rate)

    return image


def rotate_perspective(img, alphaX, alphaY):
    h, w = img.shape[:2]

    alphaX = alphaX / (180.0/math.pi)
    alphaY =  alphaY / (180.0/math.pi)

    dist=h
    f=h

    # Projection 2D -> 3D matrix
    A1= np.array([[1, 0, -w/2],
                  [0, 1, -h/2],
                  [0, 0, 0],
                  [0, 0, 1]])

    # Rotation matrices around the X axis
    Rx = np.array([[1, 0, 0, 0],
                  [0, math.cos(alphaX), -math.sin(alphaX), 0],
                  [0, math.sin(alphaX),  math.cos(alphaX), 0],
                  [0, 0, 0, 1]])
    
    # Rotation matrices around the Y axis
    Ry = np.array([[math.cos(alphaY), 0, math.sin(alphaY), 0],
                  [0, 1, 0, 0],
                  [-math.sin(alphaY), 0, math.cos(alphaY), 0],
                  [0, 0, 0, 1]])

    # Translation matrix on the Z axis 
    T = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, dist], 
                  [0, 0, 0, 1]])

    # Camera Intrisecs matrix 3D -> 2D
    A2 = np.array([[f, 0, w, 0], # w/2
                   [0, f, h, 0],  # h/2
                   [0, 0,   1, 0]])

    transfo = A2 @ (T @ (Ry @ (Rx @ A1)))

    img2 = np.zeros((h*2,w*2,4))
    img2 = cv2.warpPerspective(img, transfo,(img2.shape[1], img2.shape[0]), cv2.INTER_LINEAR )

    return img2

# create one image for training dataset
def make_dataset_image(bg_list, class_images, amount):
    '''
        bg_list - list of background images
        class_images - list of class images
        class_names - list of class names
        amount - number of class images to overlay
    '''

    annotation = ''

    bg_id = random.randrange(0, bg_amount)
    result_img = bg_list[bg_id].copy()

    # modify background image
    result_img = adjust_contrast_brightness(result_img, np.random.uniform(0.8, 1.5), random.randrange(-30, 50))
    result_img = change_color(result_img, (random.randrange(-10, 10),
                                             random.randrange(-10, 10),
                                             random.randrange(-10, 10)))

    img_height, img_width = result_img.shape[:2]

    for i in range(amount):
        # get class image
        class_id = random.randrange(0, class_amount)
        class_img = class_images[class_id]

        # modify class image
        class_img = rotate_image(class_img, random.randrange(-180, 180))
        class_img = rotate_perspective(class_img, random.randrange(-50, 50),random.randrange(-50, 50))
        class_img = cut_transparent_borders(class_img) 
        class_img = change_brightness(class_img, random.randrange(-30, 30))
        class_img = adjust_contrast_brightness(class_img, np.random.uniform(0.8, 1.5), random.randrange(-30, 50))
        class_img = change_color(class_img, (random.randrange(-10, 10),
                                             random.randrange(-10, 10),
                                             random.randrange(-10, 10)))
        
        class_img = random_transparent_decay(class_img, 0.1)


        # overlay class image to background
        bbox, result_img = random_overlay_transparent(result_img, class_img, 0.1, 0.5)

        x, y, w, h = bbox
        
        # draw bbox
        #cv2.rectangle(result_img,(int(x-w/2), int(y-h/2)),(int(x+w/2), int(y+h/2)),(0,255,0),3)

        x = x/img_width
        y = y/img_height
        w = w/img_width
        h = h/img_height

        annotation += f'{class_id} {x} {y} {w} {h}\n'

    return result_img, annotation

# generate set of images, place it to specific folders, create header files
def build_dataset(bg_list, class_images, class_names, train_size,  amount, output_path):
    '''
    Generate dataset from class images
    '''
    train_size = np.clip(train_size,0.0,1.0)

    # define folders for train and validation part
    train_folder = os.path.join(output_path, 'train')
    val_folder = os.path.join(output_path, 'val')

    # create folders
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)     
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)    
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)     
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)   


    # generate images in cycle
    for i in tqdm.tqdm(range(amount)):

        # generate image and annotation
        image, annotation = make_dataset_image(bg_list, class_images, random.randint(1,4))
        
        # define target folder
        if i<amount*train_size:
            target_folder = train_folder
        else:
            target_folder = val_folder

        # put files in target folder
        cv2.imwrite(f'{target_folder}/images/{i}.png', image)
        with open(f"{target_folder}/labels/{i}.txt", "w") as text_file:
            text_file.write(annotation)

    # create classes txt file
    with open(f"{output_path}/classes.txt", "w") as text_file:
        text_file.writelines([x+'\n' for x in class_names])


    # create data.yaml file
    yaml = f'train: {os.path.abspath(train_folder)}\n'
    yaml+= f'val: {os.path.abspath(val_folder)}\n'
    yaml+= f'nc: {len(class_names)}\n'
    yaml+= f'names: {[ x for x in class_names]}\n'
    with open(f"{output_path}/dataset.yaml", "w") as text_file:
        text_file.write(yaml)


build_dataset(bg_list, class_images, class_names, 
              train_size=0.9, 
              amount=10, 
              output_path='C:\\Users\\artem\\Desktop\\work\\puncher\\cards_names\\dataset')
