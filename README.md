The objective of this assignment is to create an automated image stitching system that produces seamless panoramic images from either a set of images with overlapping fields of view or frames extracted from a short video. The system will take images/video frames as input and combine them into a single unified panorama spanning the full horizontal field of view captured. The output is a high-resolution panoramic image stitching the aligned images/frames together into a wide vista

This code performs two tasks, and the final panorama image will be created at the output_path location.

– For the first task, input_path will contain the location of the folder containing the
images.

python3 main.py 1 input_path output_path

– For the second part, input_path will contain the location of the video file.

python3 main.py 2 input_path output_path
