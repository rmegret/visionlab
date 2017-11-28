# Cheatsheet: skimage and related functions

## NumPy

```python
import numpy as np
```

__Basic array operations__

```python
A = np.random.random([h,w])     # (h,w) array with random values in [0,1]
A = np.zeros((h,w))             # (h,w) array of zeroes
A = np.ones((h,w))              # (h,w) array of ones
A = np.array([[1,2,3],[4,5,6]]) # (2,3) array with custom values
A = B.copy()                    # Duplicate data structure
A+B, A-B, A*B, A/B, A**B        # Element-wise operations
```

```python
A.dtype   # Data type (uint8, float...)
A.shape   # (w,h,...)
A.size    # total number of elements
```

__Slicing__

```python
A[y,x,:]  or A[y,x,...]       # Pixel (x,y)
A[y,:,:]                      # Row y
A[:,x,:]                      # Column x
A[:,:,c]  or  A[...,c]        # Channel c
A[y1:y2,x1:x2,:]  # Rectangle [x1,x2)*[y1,y2) (x2 and y2 are excluded)
A[1:-1,1:-1,:]    # Crop: Remove 1 pixel margin around the image
A[::2,::2,:]      # Subsample: keep only pixels with a step of 2
```

__Thresholding and boolean operations__

```python
mask = A[...,0]>50                      # Array of booleans, one dim less than A
mask = (A[...,0]>50) & (A[...,0]<80))   # and (Don't forget the parentheses !)
mask = (A[...,0]>50) | (A[...,1]<30))   # or
mask = ~mask1                           # not
mask = mask1 ^ mask2                    # xor
```

Assignment, with broadcasting:

```python
A[...]=B                           # Copy values into preexisting array
A[10:60, 10:210, :] = [255, 0, 0]  # Vector gets replicated automatically
A[mask, :] = [255, 0, 0]  # Assign to each pixel A[i,j,:] where mask[i,j]==True
A[mask, :] = B[mask, :]   # Copy pixel values from B where mask[i,j]==True
```

__Reorganizing arrays__

We assume `A.shape==(h,w,3)`

```python
B=A.transpose([2,0,1])   # Permute dims to (3,w,h)
B=A.reshape((h*w,3))     # Reorganize items into new shape
B=A.ravel()              # Reshape to a flat 1D array
```

## Plotting

```python
import matplotlib.pyplot as plt
```

__Layout__

```python
fig = plt.figure() 
fig = plt.figure(figsize=(5,3)) # New figure ((w,h) size in inches)

fig, axes = plt.subplots(1,3)   # New figure with row of 3 subplots
axes[i].plot(...)               # Draw in subplot #i

fig, axes = plt.subplots(3,4)   # 3x4 matrix of subplots
axes[i,j].plot(...)             # Draw in subplot (i,j)
# or
axes=axes.ravel()               # Flatten the axes array to have simpler index
axes[k].plot(...)
```

__Basic plots__

```python
plt.plot(xs, ys, '.-')        # plot curve (xs[i],ys[i])
```

```python
plt.imshow(A, cmap='viridis')      # Graylevel image with colormap
plt.imshow(A, vmin=a, vmax=b)      # Set explicitly min and max values

skdemo.imshow_all(A, B, C)         # Draw 3 images side by side
skdemo.imshow_all(A, B, C, limits='auto')  # Each image gets its own limits
plt.tight_layout()  # Fix overlapping axes
```

```python
H,bin_edges,_ = plt.hist(A.ravel(), 100)      # Histogram of values, 100 bins

H,bin_edges = np.histogram(A.ravel(), 100)
plt.bar(bin_edges[:-1], H, width=1, align='edge')

from skimage import exposure
H, bin_centers = exposure.histogram(image)    # If `int`, bins = 0..255
plt.plot(bin_centers, H)
plt.bar(bin_centers, H, width=1)
```

__Colorbars__

```python
# Official way to add colorbar:
imhandle = plt.imshow(A)            # Need to get the handle to the drawn image
plt.colorbar(imhandle, ax=axes[1])  # Use imhandle colormap for axes[1] colorbar

# Simplified form from skdemo (use some tricks internally):
skdemo.colorbars()                   # Add colorbars to all axes in current figure
skdemo.colorbars(axes[1])            # Add colorbar to axes[1]
skdemo.colorbars([axes[0],axes[2]])  # Add colorbars to axes[0] and axes[2] only
```

__Access plot elements properties__

```python
# Tricks to change plot elements after plotting:

# All elements follow a hierarchy: figure -> axes -> images
ax = fig.axes[i]                     # Axes #i in figure `fig`
imhandle = ax.images[0]              # First image in axes `ax`
ax = imhandle.axes                   # Axes in which imhandle is drawn

# Current figure/axes/image
fig = plt.gcf();    axes = plt.gca();    imhandle = plt.gci();   # Get
plt.scf(fig);       plt.sca(ax);         plt.sci(imhandle);      # Set
```

```python
# Current figure properties
plt.suptitle('Figure title')

# Current axes properties
plt.title('Axes title')
plt.xlabel('X axis name'); plt.ylabel('Y axis name')
plt.xticks([]); plt.yticks([])       # Remove ticks from current axes
plt.xticks(A.shape[1]); plt.yticks(A.shape[0])  # Force integer ticks

# Current image properties
plt.cmap('viridis')         # Change colormap
plt.clim(0,1)               # Force min/max values
```

Note: Most properties can also be accessed with object oriented API instead. For instance: `ax.set_label('X axis name')` or `imhandle.set_cmap('viridis')`

### Image manipulation

```python
from skimage import data, io, color, exposure, filters, \
                    transform, morphology, measure
from scipy import ndimage as ndi
```

__I/O, Data types__

```python
from skimage import data, io
image = data.chelsea()
balloon = io.imread('../images/balloon.jpg')

from skimage import img_as_float, img_as_ubyte
image_float = img_as_float(image)     # Automatically convert [0..255] to [0,1]
image_ubyte = img_as_ubyte(image)     # Automatically convert [0,1] to [0..255]
image_float = image.astype(np.float)  # Keep same values but change dtype
image.dtype, image.min(), image.max() # Data type and value range

from skimage import transform
scaled  = transform.rescale(image, 0.25)    # Scale factor
resized = transform.resize(image, (100,80)) # Resize to given shape
```

__Simple processing__

```python
image+10, image-10                  # Brighten, Darken
image*1.3, (image-128)*1.3+128      # Increase contrast (keep 0 or 128 invariant)
high_contrast=(image-a)/(b-a)*255   # Maps [a,b] to [0,255]

from skimage import exposure
high_contrast = exposure.rescale_intensity(image, 
                     in_range=(a,b), out_range=(0,255))
```

__Color conversion__

```python
from skimage import color
gray = color.rgb2gray(balloon)
lab  = color.rgb2lab(balloon)
hsv  = color.rgb2hsv(balloon)
```

__Filtering__

```python
image = img_as_float(image)    # Make sure images are float before processing

from scipy import ndimage as ndi
filtered = ndi.convolve(image, mean_kernel) # 2d convolution, generic kernel

from skimage import filters
smooth = filters.gaussian(image, sigma) # 2d convolution, gaussian
gradient_magnitude = filters.sobel(image)
gradient_magnitude = filters.sobel(image, )

from skimage import morphology
selem = morphology.disk(1)  # structuring element
median  = filters.rank.median(pixelated, selem)
eroded  = morphology.erosion(image, selem)
dilated = morphology.dilation(image, selem)
opened  = morphology.opening(image, selem)
closed  = morphology.closing(image, selem)
```

__Analysis__

```python
from scipy import ndimage as ndi
label_image, count = ndi.label(mask)

from skimage import measure
props = measure.regionprops(label_image)
centroids = np.array( [prop.centroid for prop in props] )
areas     = np.array( [prop.area for prop in props] )
```


