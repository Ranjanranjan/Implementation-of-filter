# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import necessary libraries (cv2, NumPy, Matplotlib) for image loading, filtering, and visualization.

### Step 2: 
Load the image using cv2.imread() and convert it to RGB format using cv2.cvtColor() for proper display in Matplotlib.

### Step 3: 
Apply different filters:
1. Averaging Filter: Define an averaging kernel using np.ones() and apply it to the image using cv2.filter2D().
2. Weighted Averaging Filter: Define a weighted kernel (e.g., 3x3 Gaussian-like) and apply it with cv2.filter2D().
3. Gaussian Filter: Use cv2.GaussianBlur() to apply Gaussian blur.
4. Median Filter: Use cv2.medianBlur() to reduce noise.
5. Laplacian Operator: Use cv2.Laplacian() to apply edge detection.
    

### Step 4: 
Display each filtered image using plt.subplot() and plt.imshow() for side-by-side comparison of the original and processed images.

### Step 5: 
Save or show the images using plt.show() after applying each filter to visualize the effects of smoothing and sharpening.


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7, 3))
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
```
<img width="322" alt="Screenshot 2024-10-21 at 11 14 37 AM" src="https://github.com/user-attachments/assets/2ad76653-16a5-4365-8501-b0429c0ff8f2">

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(7, 3))
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()
```
<img width="311" alt="Screenshot 2024-10-21 at 11 15 07 AM" src="https://github.com/user-attachments/assets/180b668b-1a27-4e6c-85fb-198a4e5b45f6">

ii) Using Weighted Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(7, 3))
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
<img width="318" alt="Screenshot 2024-10-21 at 11 15 37 AM" src="https://github.com/user-attachments/assets/ac7bc709-7969-492c-af90-733e42904dc5">


iii) Using Gaussian Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)

plt.figure(figsize=(7, 3))
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
<img width="318" alt="Screenshot 2024-10-21 at 11 16 11 AM" src="https://github.com/user-attachments/assets/ab084cf3-6013-41f2-a261-d06b4af643a5">


iv)Using Median Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
median_blur = cv2.medianBlur(image2, 11)

plt.figure(figsize=(7, 3))
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()

```
<img width="314" alt="Screenshot 2024-10-21 at 11 20 52 AM" src="https://github.com/user-attachments/assets/7ed853e3-9fa9-42b4-babb-1edace4a0e5f">

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
sharpened_image = cv2.filter2D(image2, -1, kernel1)

plt.figure(figsize=(7, 3))
plt.imshow(sharpened_image)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()
```
<img width="331" alt="Screenshot 2024-10-21 at 11 21 26 AM" src="https://github.com/user-attachments/assets/e1f70701-8bb1-49bb-ac15-584f7cdb2f85">


ii) Using Laplacian Operator
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("/content/uiop.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
laplacian = cv2.Laplacian(image2, cv2.CV_64F)

plt.figure(figsize=(7, 3))
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()


```
<img width="920" alt="Screenshot 2024-10-21 at 11 21 58 AM" src="https://github.com/user-attachments/assets/701714db-b313-414d-8394-816c81000ec4">


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
