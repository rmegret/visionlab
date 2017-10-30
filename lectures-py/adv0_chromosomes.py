
# coding: utf-8




# <codecell>

from __future__ import division, print_function
%matplotlib inline   


# <markdowncell>
# # Measuring chromatin fluorescence
# 
# Goal: we want to quantify the amount of a particular protein (red
# fluorescence) localized on the centromeres (green) versus the rest of the
# chromosome (blue).
# 
# <img src="../images/chromosomes.jpg" width="550px"/>
# 
# The main challenge here is the uneven illumination, which makes isolating the
# chromosomes a struggle.



# <codecell>

import numpy as np
from matplotlib import cm, pyplot as plt
import skdemo
plt.rcParams['image.cmap'] = 'cubehelix'
plt.rcParams['image.interpolation'] = 'none'   



# <codecell>

from skimage import io
image = io.imread('../images/chromosomes.tif')
skdemo.imshow_with_histogram(image);   


# <markdowncell>
# Let's separate the channels so we can work on each individually.



# <codecell>

protein, centromeres, chromosomes = image.transpose((2, 0, 1))   


# <markdowncell>
# Getting the centromeres is easy because the signal is so clean:



# <codecell>

from skimage.filter import threshold_otsu
centromeres_binary = centromeres > threshold_otsu(centromeres)
skdemo.imshow_all(centromeres, centromeres_binary)   


# <markdowncell>
# But getting the chromosomes is not so easy:



# <codecell>

chromosomes_binary = chromosomes > threshold_otsu(chromosomes)
skdemo.imshow_all(chromosomes, chromosomes_binary, cmap='gray')   


# <markdowncell>
# Let's try using an adaptive threshold:



# <codecell>

from skimage.filter import threshold_adaptive
chromosomes_adapt = threshold_adaptive(chromosomes, block_size=51)
# Question: how did I choose this block size?
skdemo.imshow_all(chromosomes, chromosomes_adapt)   


# <markdowncell>
# Not only is the uneven illumination a problem, but there seem to be some
# artifacts due to the illumination pattern!
# 
# **Exercise:** Can you think of a way to fix this?
# 
# (Hint: in addition to everything you've learned so far, check out
# [`skimage.morphology.remove_small_objects`](http://scikit-image.org/docs/dev/a
# pi/skimage.morphology.html#skimage.morphology.remove_small_objects))



# <codecell>

   


# <markdowncell>
# Now that we have the centromeres and the chromosomes, it's time to do the
# science: get the distribution of intensities in the red channel using both
# centromere and chromosome locations.



# <codecell>

# Replace "None" below with the right expressions!
centromere_intensities = None
chromosome_intensities = None
all_intensities = np.concatenate((centromere_intensities,
                                  chromosome_intensities))
minint = np.min(all_intensities)
maxint = np.max(all_intensities)
bins = np.linspace(minint, maxint, 100)
plt.hist(centromere_intensities, bins=bins, color='blue',
         alpha=0.5, label='centromeres')
plt.hist(chromosome_intensities, bins=bins, color='orange',
         alpha=0.5, label='chromosomes')
plt.legend(loc='upper right')
plt.show()   


# <markdowncell>
# ---
# 
# <div style="height: 400px;"></div>



# <codecell>

%reload_ext load_style
%load_style ../themes/tutorial.css   
