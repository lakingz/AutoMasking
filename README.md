# Automatically Masking Cartridge Case Images

This package contains python functions to create automatically masking for cartridge case images

## Target to mask

1. The breech-face Impression (red)
2. The aperture shear (green) incomplete
3. The firing pin impression (purple)
4. The firing pin drag (light blue)
5. The direction of the firing pin drag (blue arrow)

### Dependencies

numpy '1.26.3'   
OpenCV '4.9.0'

### Executing program

replace 'images/Picturetrain.png' with 'file_directory' for parameter image_path and then execute the remaining code.

## Descriptions

The package contains three files:

1_bf_masking.py 
(Automatically masking for task 1. The breech-face Impression (red))

3_4_5_fp_masking.py  

(Automatically masking for task 3. The firing pin impression (purple); 4. The firing pin drag (light blue); 5. The direction of the firing pin drag (blue arrow))

combine.py   
(A combination of two above functions. It creates a masked image at 'images/' folder)

## Note

The task 2. The aperture shear (green) is incomplete given a limited training image. With larger training set it is possible to train a suitable filter just for this task with scipy and cv2.filter2D.  

## Authors

ex. Ankai Liu 
ex. [@Ankai Liu](https://github.com/lakingz/AutoMasking)

## Version History

* 0.1
    * Initial Release

## License

N/A
## Acknowledgments

1_bf_masking.py is partially inspired by
* Tai, Xiao Hui; Eddy, William F. (2017). A Fully Automatic Method for Comparing Cartridge Case Images,. Journal of Forensic Sciences, (), –. doi:10.1111/1556-4029.13577
