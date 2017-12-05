
# coding: utf-8



# <markdowncell>
# ## Counting grains and bubbles
# 
# This Scanning Element Microscopy image shows a glass sample
# (light gray matrix) with some bubbles (black) and unmolten
# sand grains (dark gray). We wish to determine the fraction
# of the sample covered by these three phases,
# and to estimate the number of sand grains and bubbles,
# their average sizes, etc.


# <markdowncell>
# ### Loading the slide


# <codecell>

%matplotlib inline
import numpy as np

import matplotlib
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.rcParams['figure.figsize'] = (10, 7)   


# <codecell>

from skimage import io
from matplotlib import pyplot as plt

img = io.imread('../images/bubbles.jpg')

plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest');   


# <markdowncell>
# ### Remove banner


# <codecell>

img_clean = img[:880, :]
plt.imshow(img_clean, cmap=plt.cm.gray, interpolation='nearest');   


# <markdowncell>
# ### Filter to get rid of speckles


# <codecell>

#import scipy.ndimage as ndi
from skimage.filters import median
from skimage.morphology import rectangle

#img_med = ndi.median_filter(img_clean, size=5)
img_med = median(img_clean, rectangle(5,5))
plt.imshow(img_med, cmap=plt.cm.gray, interpolation='nearest');   


# <markdowncell>
# ### Find threshold values


# <codecell>

plt.figure(figsize=(8,3))
plt.hist(img_med.flatten(), bins=40, range=(0, 150));   


# <markdowncell>
# ### Separate layers by thresholding


# <codecell>

bubbles = (img_med <= 50)
sand = (img_med > 50) & (img_med <= 120)
glass = (img_med > 120)

def plot_images(cmap=plt.cm.gray):
    for n, (name, image) in \
        enumerate([('Original', img_med),
                   ('Bubbles', bubbles),
                   ('Sand', sand),
                   ('Glass', glass)]):
    
        plt.subplot(2, 2, n + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(name)
        plt.axis('off')
        
plot_images();   


# <markdowncell>
# ### Visualise layers


# <codecell>

def plot_color_overlay():
    all_layers = np.zeros((img_clean.shape[0],
                           img_clean.shape[1], 3)) # Color image

    # You shouldn't run this if bubbles isn't a mask
    # -- otherwise, fancy indexing instead of masking
    assert(bubbles.dtype == np.bool)
    
    all_layers[bubbles] = (1, 0, 0)
    all_layers[sand] = (1, 1, 0)
    all_layers[glass] = (0, 0, 1)

    plt.imshow(all_layers)

plot_color_overlay()   


# <markdowncell>
# ### Clean up shapes found


# <codecell>

from skimage.morphology import opening, closing

for img in (sand, bubbles, glass):
    #img[:] = ndi.binary_opening(img, np.ones((5, 5)))
    #img[:] = ndi.binary_closing(img, np.ones((5, 5)))
    img[:] = opening(img, np.ones((5, 5)))
    img[:] = closing(img, np.ones((5, 5)))
    
plot_images()   


# <markdowncell>
# ### Label connected components


# <codecell>

from skimage.measure import label

# Convert to int so we can store the labels
bubbles = bubbles.astype(int)
sand = sand.astype(int)
glass = glass.astype(int)

obj_areas={}

for name, img in [('Sand', sand),
                  ('Bubbles', bubbles),
                  ('Glass', glass)]:
    #labels, count = ndi.label(img) # Scipy
    labels, count = label(img, return_num=True) # skimage
    print('%s regions found in %s' % (count, name))
    img[:] = labels
    
    obj_areas[name] = [np.sum(labels == i) for \
                 i in range(1, labels.max())]
    print("Mean obj area %d" % np.mean(obj_areas[name]))

plot_images(cmap=plt.cm.spectral)   


# <codecell>

count,bins=np.histogram(np.sqrt(obj_areas['Bubbles']), 10)
#bins=(bins[0:-1]+bins[1:])/2
#plt.bar#

plt.figure(figsize=(6,3))
plt.bar(bins[0:-1],count,width=bins[1:]-bins[0:-1], align='edge')
plt.xlabel('size')
plt.ylabel('count')
plt.title('Bubbles')

#df=pd.DataFrame(data={'est_size':bins.T,'count':count.T},columns=['est_size','count'])
#df   
