## Quantitative results

Model       Parameters      PSNR        SSIM        Time

Bicubic     0               24.34       0.6398      N/A
SRCNN       20,099          23.90       0.6381      0.415 ms
ESPCN       37,200          22.91       0.5824      0.295 ms

## Observations

- Bicubic got a higher PSNR/SSIM 

- SRCNN achieved better reconstruction quality than ESPCN due to refining images in high-res space, but requires more computation

- ESPCN provides faster inference by performing feature extraction in low-res space and using pixelshuffle for upscaling.

- Neural networks could achieve higher scores with larger datasets, deeper architectures, longer training and more advanced loss functions