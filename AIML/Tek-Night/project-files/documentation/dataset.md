# Dataset source
```
DIV2K Dataset
```

- 800 high resolution images were used from the DIV2K datset.

The images were split into train, validation and test accordingly:

Table:

**Split**	       **Images**
Train	            700
Validation	        50
Test	            50

### Degradation pipeline:

HR image
    |
    |
Crop 128x128
    |
    |
Downsample x4
(added gaussian blur, bicubic downsampling,
jpeg compression, gaussian noise)
    |
    |
LR image
(32x32)