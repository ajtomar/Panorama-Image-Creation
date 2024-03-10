import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
import math
import sys
import cv2
import random
import shutil

def print_img(img,title):
    plt.imshow(img)
    plt.title(title)
    plt.show()

def print_img_subplots(img1, img2,title1,title2):
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.show()

def convolve(input_img, conv_kernel, use_padding):
    if use_padding:
        pad_height, pad_width = conv_kernel.shape[0] // 2, conv_kernel.shape[1] // 2
        input_img = np.pad(input_img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    result_height = input_img.shape[0] - conv_kernel.shape[0] + 1
    result_width = input_img.shape[1] - conv_kernel.shape[1] + 1
    windows_matrix = np.lib.stride_tricks.sliding_window_view(input_img, conv_kernel.shape).reshape(result_height*result_width,-1)
    flattened_conv_kernel = conv_kernel.flatten()
    result = np.dot(windows_matrix, flattened_conv_kernel)
    result = result.reshape(result_height, result_width)
    return result

def plot_extremas(img,key_pts):
    temp_img = img.copy()
    size = len(key_pts)
    print('Extremas Size: ', size)
    cent_color = (255, 0, 0)
    rad = 5
    for i in key_pts:
        cv2.circle(temp_img, (i[1], i[0]), rad, cent_color, -1)
    print_img_subplots(img,temp_img,'Original Image','Image with keypoints')

def exp_val(x,y,c_1,c_0,sigma):
    return np.exp(-((x - c_1)**2 + (y - c_0)**2) / (2 * sigma**2))

def gausian_discriptor(size, sigma,center):
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    exp_values = exp_val(x,y,center[1],center[0],sigma)
    return exp_values / np.sum(exp_values)

def create_gaussian(sigma):
    size = 15
    center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    exp_values = exp_val(x,y,center,center,sigma)
    return exp_values / np.sum(exp_values)

def upsampling_image(image):
    row, col = image.shape
    new_img = np.zeros((2*row, 2*col))
    
    for i in range(0, row):
        for j in range(0, col):
            new_img[2*i][2*j] = image[i][j]

    new_sigma = math.sqrt(max((1.6 ** 2) - ((2 * 0.5) ** 2), 0.01))
    kernel = create_gaussian(new_sigma)
    img = convolve(new_img, kernel, True)
    return img

def downsample_image(image):
    row, col = image.shape
    new_img = np.zeros((row//2, col//2))
    
    for i in range(0, row//2):
        for j in range(0, col//2):
            new_img[i][j] = image[2*i][2*j]

    return new_img

def check(arr1, arr2, arr3):
    c = arr2[1, 1]
    condition1 = np.all(arr1 >= c) and np.all(arr3 >= c) and np.all(arr2[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 0, 1, 2]] >= c)
    condition2 = np.all(arr1 <= c) and np.all(arr3 <= c) and np.all(arr2[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 0, 1, 2]] <= c)
    return condition1 or condition2

def print_DOG_img(DOG):
    track=1
    print('Print DOGs')
    for i in range(len(DOG)):
    #     print(DOG[i])
    #     print(i)
        # plt.imshow(DOG[i],cmap='gray')
        # plt.title('Image')
        # plt.axis('off')
        # plt.show()
        if (track%5==0):
            print('*************************************************************************')
        track+=1    

def plot_extremas(img,key_pts):
    temp_img = img.copy()
    size = len(key_pts)
    print('Extremas Size: ', size)
    cent_color = (255, 0, 0)
    rad = 5
    for i in key_pts:
        cv2.circle(temp_img, (i[1], i[0]), rad, cent_color, -1)
    print_img_subplots(img,temp_img,'Original Image','Image with keypoints')

def calculate_hessian_matrix_elements(location):
    return location[1, 2] + location[1, 0] - 2 * location[1, 1], location[0, 1] + location[2, 1] - 2 * location[1, 1], 0.25 * (location[0, 2] + location[2, 0] - location[0, 0] - location[2, 2])

def is_not_edge_like(location):
    dxx, dyy, dxy = calculate_hessian_matrix_elements(location)
    edge_threshold = 10
    return (edge_threshold + 1) * 2 * (dxx * dyy - dxy * 2) > edge_threshold * (dxx + dyy) ** 2

def edge_points_removal(location):
    return is_not_edge_like(location)

def all_max_angels(one_point_bin):
    original=one_point_bin.copy()
    one_point_bin=one_point_bin/np.max(one_point_bin)
    # display_hist(one_point_bin)
    max_pos=np.where(one_point_bin>0.8)[0]
    # print(max_pos)
    principal_angels=[]
    for i in max_pos:
        p1=(i-1+36)%36
        p3=(i+1+36)%36
        t1=(10*p1)+5
        t2=(10*i)+5
        t3=(10*p3)+5
        x=[t1,t2,t3]
        y=[original[p1],original[i],original[p3]]
        A = np.vstack([np.array(x)**2, x, np.ones(len(x))]).T
        coefficients = np.linalg.solve(A, y)
        final_t=-(coefficients[1])/(2*coefficients[0])
        if(final_t<=360 and final_t>=0):
            principal_angels.append(final_t)
        # print(coefficients,final_t)
    return principal_angels

def is_within_boundaries(y, x, h, w):
    return not (y - 8 < 0 or y + 10 >= h or x - 8 < 0 or x + 10 >= w)

def calculate_orientation_and_magnitude(dx, dy):
    angel = np.degrees(np.arctan2(dy, dx)) % 360
    magnitude = np.sqrt((dx**2) + (dy**2))
    return angel, magnitude

def create_orientation_histogram(angel, magnitude):
    one_point_bin = np.zeros(36)
    for m in range(16):
        for n in range(16):
            one_point_bin[int(angel[m,n] // 10)] += magnitude[m,n]
    return one_point_bin

def generate_sub_histogram(magnitude, angel, sub_gauss, angels):
    final_bin = np.zeros(8)
    sub_magnitude = (magnitude.copy())*(sub_gauss)
    sub_angel = angel.copy()-angels
    sub_angel[sub_angel < 0] += 360
    for p in range(4):
        for q in range(4):
            final_bin[int(sub_angel[p, q] // 45)] += sub_magnitude[p, q]
    return final_bin

def create_descriptor(gaussian_imgs, p, sigma_diff, octave_no, i, h, w, all_bins, big_gauss):
    y, x = p
    sigma = np.sum(sigma_diff[:i+1]) * (2 ** octave_no) * 1.5
    gaus = gausian_discriptor(16, sigma, np.array([7, 7]))

    dx = convolve(gaussian_imgs[i][y - 7:y + 9, x - 8:x + 10], np.array([[-1, 0, 1]]), False)
    dy = convolve(gaussian_imgs[i][y - 8:y + 10, x - 7:x + 9], np.array([[-1, 0, 1]]).T, False)
    
    angel, magnitude = calculate_orientation_and_magnitude(dx, dy)
    new_gaus=magnitude*gaus
    one_point_bin = create_orientation_histogram(angel, new_gaus)
    
    principal_angels = all_max_angels(one_point_bin)
    for angels in principal_angels:
        disk = np.array([])
        for m in range(0, 16, 4):
            for n in range(0, 16, 4):
                sub_magnitude = magnitude[m:m + 4, n:n + 4]
                sub_angel = angel[m:m + 4, n:n + 4]
                final_bin = generate_sub_histogram(sub_magnitude, sub_angel, big_gauss[m:m + 4, n:n + 4], angels)
                disk = np.concatenate((disk, final_bin))

        x_original = math.floor(x * (2 ** (octave_no - 1)))
        y_original = math.floor(y * (2 ** (octave_no - 1)))
        mapped_points = (y_original, x_original)
        
        if mapped_points not in all_bins:
            all_bins[mapped_points] = []
        all_bins[mapped_points].append(disk)


def discriptor(p, gaussian_imgs, octave_no, sigma_diff):
    all_bins = {}
    p_c = np.array([7, 7])
    big_gauss = gausian_discriptor(16, 8, p_c)
    h, w = gaussian_imgs[0].shape
    sigma_diff = [element ** 2 for element in sigma_diff]

    for i, points in p.items():
        for point in points:
            y, x = point
            if not is_within_boundaries(y, x, h, w):
                continue
            create_descriptor(gaussian_imgs, point, sigma_diff, octave_no, i, h, w, all_bins, big_gauss)

    return all_bins

def merge(dict1, dict2):
    merged_dict = {}
    for key in dict1.keys() | dict2.keys():
        merged_dict[key] = dict1.get(key, []) + dict2.get(key, [])
    return merged_dict

def cal_sigma_val(sigma,size):
    sigma_val = []
    val=0
    for i in range(size):
        temp = (2**(i/3))*sigma
        sigma_val.append(np.sqrt(temp**2-val**2))
        val =temp
    return sigma_val

def is_key_point(dog, i, j, k, leave, threshold=0.0025):
    # Check if the center element is above the threshold
    if dog[i + 1, j, k] <= threshold:
        return False
    
    # Apply edge points removal criteria
    if not edge_points_removal(dog[i + 1, j - leave:j + leave + 1, k - leave:k + leave + 1]):
        return False
    
    # Check if the center element is a local extremum
    if not check(dog[i, j - leave:j + leave + 1, k - leave:k + leave + 1], 
                 dog[i + 1, j - leave:j + leave + 1, k - leave:k + leave + 1], 
                 dog[i + 2, j - leave:j + leave + 1, k - leave:k + leave + 1]):
        return False
    
    return True

def points(dog, octave_number, s, unique_key_points):
    leave = s // 2
    keys_in_octave = {}
    
    for i in range(len(dog) - 2):
        for j in range(leave, dog[0].shape[0] - leave):
            for k in range(leave, dog[0].shape[1] - leave):
                if is_key_point(dog, i, j, k, leave):
                    unmapped_point = [j, k]
                    mapped_point = (math.floor(j * (2 ** (octave_number - 1))), 
                                    math.floor(k * (2 ** (octave_number - 1))))
                    keys_in_octave.setdefault(i + 1, []).append(unmapped_point)
                    unique_key_points.add(mapped_point)
                    
    return keys_in_octave

def find_keypoints_descriptors(img,num_octave,samp_per_octave):
#     print_img_subplots(img,gray_img,'Original Image', 'Gray Image' )
    up_img=upsampling_image(img)
    image=up_img.copy()
#     print_img(image_temp,'Upsampled and Gauss Image')
    g_size=5
    sigma=1.6
    all_key_points=[]
    u_kpts=set()
    g_disc={}
    sigma_values = cal_sigma_val(sigma,samp_per_octave+3)
    for k in range(num_octave):
        DOG_images = []
        gaussian_img = []
        gaussian_img.append(image)
        prev = image
        print('Octave no: ',k+1)
#         print_img(image,f'Base Gauss Image {k+1}*Sigma')
        for i in range(1,samp_per_octave+3):
#             print('Level: ',i)
#             new_sigma = (2**((k*samp_per_octave+i)/samp_per_octave))*sigma
            new_sigma = (2**(i/samp_per_octave))*sigma
#             kernel = create_gaussian(5,new_sigma)
#             image_temp = convolution(image, kernel)
            kernel = create_gaussian(new_sigma)
#             print('find_key')
#             print(image.shape)
            image_temp = convolve(image, kernel, True)
            gaussian_img.append(image_temp)
            if i==samp_per_octave:
                sigma_img = image_temp
#             print(image_temp.shape,prev.shape)
            DOG_images.append(np.array(image_temp-prev))
            prev=image_temp

        DOG_images = np.array(DOG_images)
        gaussian_img=np.array(gaussian_img)
        # print('DOG len:', len(DOG_images))
        curr_octave_kpts = points(DOG_images,k,3,u_kpts)
        # print('curr_octave_kpts len:', len(curr_octave_kpts))
        l_disc=discriptor(curr_octave_kpts,gaussian_img,k,sigma_values)
        # print('l_disc len:', len(l_disc))
        g_disc=merge(g_disc,l_disc)
        image = downsample_image(sigma_img)
        # break

    return u_kpts,g_disc



def save_frame_at_index(video_capture, frame_index, output_dir, frame_number):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = video_capture.read()
    if success:
        filename = f"frame_{frame_number}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)

def capture_frames_at_times(video_file_path, output_directory, total_frames=5):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    video_capture = cv2.VideoCapture(video_file_path)
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_file_path}")
        return

    try:
        total_vid_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_vid_frames - 1, total_frames, dtype=int)

        for count, frame_idx in enumerate(frame_indices, start=1):
            save_frame_at_index(video_capture, frame_idx, output_directory, count)
    finally:
        video_capture.release()





if int(sys.argv[1]) == 1:
    folder_path = sys.argv[2]
elif int(sys.argv[1]) == 2:
    folder_path = os.path.join(sys.argv[3], 'Images_ss')  
    video_path = sys.argv[2]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    # os.makedirs(folder_path)  # Create new folder
output_path = sys.argv[3]

if int(sys.argv[1])==2:
    capture_frames_at_times(video_path, folder_path)
print('Data Loaded')

images_rgb = []
images = os.listdir(folder_path)
images_file = np.sort(images)
num_images = len(images_file)
# print('images: ',images_file)
# print('folder path: ',folder_path)
disc = []

for curr_img in images_file:
    # if curr_img!= '3.jpg':
    #     continue
    image_path = os.path.join(folder_path,curr_img)
    image = cv2.imread(image_path)
    t_img = image.copy()
    rgb_image = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
    images_rgb.append(rgb_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=image/255
    num_octave = 6
    samp_per_octave = 3
    
    img = image.copy()
    key_pts,d = find_keypoints_descriptors(img,num_octave,samp_per_octave)
    disc.append(d)
    plot_extremas(rgb_image,key_pts)


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

keypoints=[]
discriptors=[]
for i in disc:
    temp_points=[]
    temp_disc=[]
    for point,dis in i.items():
        for j in dis:
            temp_points.append(point)
            temp_disc.append(j)
    keypoints.append(temp_points)
    discriptors.append(temp_disc)

for i in range(len(images_file)):
    print(len(keypoints[i]),len(discriptors[i]))

images = []

file_list = os.listdir(folder_path)
file_list=np.sort(file_list)
for file_name in file_list:    
    image_path = os.path.join(folder_path, file_name)
    bgr_image = cv2.imread(image_path)
    height, width = bgr_image.shape[:2]
    bgr_image = cv2.resize(bgr_image, (width, height))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    images.append(rgb_image)

def find_best_match(descriptor1, descriptors2, point2, w2,best_distance = float('inf'),second_best_distance = float('inf'), best_match_index=None):
     
    val =[]
    for j, descriptor2 in enumerate(descriptors2):
        if (point2[j][1] < w2 // 2):
            if (euclidean_distance(descriptor1, descriptor2)) < best_distance:
                second_best_distance = best_distance
                val.append(best_match_index)
                best_distance = euclidean_distance(descriptor1, descriptor2)
                best_match_index = j
            elif euclidean_distance(descriptor1, descriptor2) < second_best_distance:
                second_best_distance = euclidean_distance(descriptor1, descriptor2)

    return best_match_index, best_distance, second_best_distance

def is_valid_point(point, width):
    return point[1] > width // 2


def is_good_match(best_distance, second_best_distance, threshold):
    return best_distance <= threshold * second_best_distance

def match_keypoints(descriptors1, descriptors2, point_list1, point_list2, shape1, shape2, threshold=0.8):
    matches = []
    h1, w1 = shape1
    h2, w2 = shape2

    for i, descriptor1 in enumerate(descriptors1):
        if is_valid_point(point_list1[i], w1):
            best_match_index, best_distance, second_best_distance = find_best_match(descriptor1, descriptors2, point_list2, w2)
#             best_match_index, best_distance, second_best_distance = find_best_match(descriptor1, descriptors2, point_list2, w2, None, float('inf'), float('inf') )

            if is_good_match(best_distance, second_best_distance, threshold):
                matches.append((i, best_match_index))

    return matches

all_matches = []
for i in range(num_images-1):
    matches = match_keypoints(discriptors[i], discriptors[i+1], keypoints[i], keypoints[i+1], images[i].shape[:2], images[i+1].shape[:2])
    all_matches.append(matches)

for i in all_matches:
    print(len(i))

def random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
images = []

file_list = os.listdir(folder_path)
file_list=np.sort(file_list)
for file_name in file_list:    
    image_path = os.path.join(folder_path, file_name)
    bgr_image = cv2.imread(image_path)
    height, width = bgr_image.shape[:2]
    bgr_image = cv2.resize(bgr_image, (width, height))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    images.append(rgb_image)

for i in range(num_images-1):
    image1 = images[i].copy()
    image2 = images[i+1].copy()

    print(i, i+1, len(all_matches[i]))      
    # p1=[]
    # p2=[]
    for match in all_matches[i]:
        kp1_idx, kp2_idx = match
        kp1 = keypoints[i][kp1_idx]
        kp2 = keypoints[i+1][kp2_idx]
        # p1.append(kp1)
        # p2.append(kp2)
        y1, x1 = kp1
        y2, x2 = kp2
        color = random_color()
        cv2.circle(image1, (int(x1), int(y1)), 10, color, -1)
        cv2.circle(image2, (int(x2), int(y2)), 10, color, -1)

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.show()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
        
    plt.show()

def find_diff(t1, t2, confidence):
    diff = t1 - t2
    d = np.linalg.norm(diff, axis=1)
    min_euc = 5
    inliers = d < min_euc
    percent_of_inlier = np.count_nonzero(inliers) / d.shape[0]
    if percent_of_inlier >= confidence:
        inlier_idx = np.where(inliers)[0]
        return inlier_idx, percent_of_inlier
    else:
        return [], 0
    
def gen_H(pts1, pts2):
    A = np.vstack([
        np.array([x, y, 1, 0, 0, 0, -px * x, -px * y, -px]) 
        for (x, y), (px, py) in zip(pts1, pts2)
        ] + [
        np.array([0, 0, 0, x, y, 1, -py * x, -py * y, -py]) 
        for (x, y), (px, py) in zip(pts1, pts2)
                ])
    
    A_trans = A.T @ A
    _, eigen_vect = np.linalg.eig(A_trans)
    H = eigen_vect[:, np.argmin(_)].reshape((3, 3))
    
    return H

def select_random_points(keypoints, num_points=4):
    new_pts = random.sample(range(len(keypoints)), num_points)
    pts = [keypoints[j] for j in new_pts]
    return pts, new_pts

def perform_homography(keypoints1, keypoints2, new_pts):
    pts1, pts2 = [keypoints1[j] for j in new_pts], [keypoints2[j] for j in new_pts]
    H = gen_H(pts1, pts2)
    return H,pts2

def update_global_points(keypoints1, keypoints2, H, new_pts, confidence):
    other_than_new = [j for j in range(len(keypoints1)) if j not in new_pts]
    other_pts_in1 = [keypoints1[j] for j in other_than_new]
    other_pts_in2 = [keypoints2[j] for j in other_than_new]
    
    array_of1 = np.array(other_pts_in1)
    array_of2 = np.array(other_pts_in2)
    result_matrix = np.hstack((array_of1, np.ones((array_of1.shape[0], 1))))
    transformed_pts = np.dot(result_matrix, H.T)
    
    transformed_pts /= transformed_pts[:, [2]]
    set_of_inliers, new_confidence = find_diff(transformed_pts[:, :2], array_of2, confidence)
    
    return set_of_inliers, new_confidence, other_than_new,other_pts_in1,other_pts_in2

def homo(keypoints1, keypoints2):
    k = 100000
    global_source = []
    global_dest = []
    confidence = 0
    mask = np.zeros(len(keypoints1))
    
    for _ in range(k):
        pts1, new_pts = select_random_points(keypoints1)
        H ,pts2= perform_homography(keypoints1, keypoints2, new_pts)
        
        set_of_inliers, conf, other_than_new ,other_pts_in1,other_pts_in2= update_global_points(keypoints1, keypoints2, H, new_pts, confidence)
        if conf > confidence:
            confidence = conf
            global_source = [other_pts_in1[idx] for idx in set_of_inliers] + pts1
            global_dest = [other_pts_in2[idx] for idx in set_of_inliers] + pts2
            mask[other_than_new] = [1 if idx in set_of_inliers else 0 for idx in other_than_new]
            mask[new_pts] = 1

    global_source = [list(arr) for arr in global_source]
    global_dest = [list(arr) for arr in global_dest]
    # print(global_source)
    H = gen_H(global_source, global_dest)
    return H, mask

def map_warped_point_to_original(H_inv, x, y):
    warped_point = np.array([x, y, 1])
    original_point_homogeneous = H_inv @ warped_point
    original_point = original_point_homogeneous[:2] / original_point_homogeneous[2]
    return np.round(original_point).astype(int)

def warpPerspective(src_image, homography_matrix, output_width, output_height):
    inv_homography = np.linalg.inv(homography_matrix)
    src_height, src_width = src_image.shape[:2]
    warp_image = np.full((output_height, output_width, 3), 255, dtype=np.uint8)

    for y in range(output_height):
        for x in range(output_width):
            src_x, src_y = map_warped_point_to_original(inv_homography, x, y)
            if 0 <= src_x < src_width and 0 <= src_y < src_height:
                warp_image[y, x] = src_image[src_y, src_x]

    return warp_image

def trans_rotation_homography(matched_points_img1, matched_points_img2, h1, w1, h2, w2):
    homography, mask = homo(matched_points_img1, matched_points_img2)
    normalized_corners = np.dot(homography, np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]]).T).T[:, :2] / np.dot(homography, np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]]).T).T[:, 2][:, np.newaxis]
    return np.dot(np.array([[1, 0, -math.floor(np.min(normalized_corners[:,0]))], [0, 1, -math.floor(np.min(normalized_corners[:,1]))], [0, 0, 1]]), homography), int(np.max(normalized_corners[:,0]) - np.min(normalized_corners[:,0])), int(np.max(normalized_corners[:,1]) - np.min(normalized_corners[:,1])), np.max(normalized_corners[:,1]), np.min(normalized_corners[:,1]), -math.floor(np.min(normalized_corners[:,0])), -math.floor(np.min(normalized_corners[:,1]))


def give_matched_points(i, x_translation, y_translation):
    matched_points_img1 = np.array([(keypoints[i][j[0]][1] + x_translation, keypoints[i][j[0]][0] + y_translation) for j in all_matches[i]])
    matched_points_img2 = np.array([(keypoints[i+1][j[1]][1], keypoints[i+1][j[1]][0]) for j in all_matches[i]])
    return matched_points_img1, matched_points_img2

def new_stritched_image_left(left_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img2):
    warp_image1 = warpPerspective(left_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1)
    extended_image = np.ones(((max(math.ceil(max_y),img2.shape[0])-min(math.floor(min_y),0)), img2.shape[1] + trans_x, 3), dtype=np.uint8) * 255
    if min_y<=0:
    
        extended_image[:warp_image1.shape[0], :warp_image1.shape[1]] = warp_image1
        extended_image[trans_y:trans_y+img2.shape[0], trans_x:trans_x+img2.shape[1]] = img2
    else:
    
        extended_image[-trans_y:-trans_y+warp_image1.shape[0], :warp_image1.shape[1]] = warp_image1
        extended_image[:img2.shape[0], trans_x:trans_x+img2.shape[1]] = img2
        trans_y = 0

    return trans_x,trans_y,extended_image

def left_stitch_new(start,end,x_translation,y_translation):
    left_img=images[start].copy()
    i=start+1
    while(i<end+1):
        img2=images[i].copy()
        matched_points_img1,matched_points_img2=give_matched_points(i,x_translation,y_translation)
        adjusted_homography_matrix,calc_warp_w1,calc_warp_h1,max_y, min_y, trans_x, trans_y=trans_rotation_homography(matched_points_img1,matched_points_img2,left_img.shape[0],left_img.shape[1],img2.shape[0],img2.shape[1])
        x_translation,y_translation,left_img=new_stritched_image_left(left_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img2)
        plt.imshow(left_img)
        plt.show()
        i+=1
    return left_img,x_translation,y_translation


def new_stritched_image_right(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img1):
    warp_image2=warpPerspective(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1)
    extended_image = np.ones(((max(math.ceil(max_y),img1.shape[0])-min(math.floor(min_y),0)), warp_image2.shape[1]- trans_x, 3), dtype=np.uint8) * 255
    if min_y<=0:
    
        extended_image[:warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
        extended_image[trans_y:trans_y+img1.shape[0], :img1.shape[1]] = img1
    else:
    
        extended_image[-trans_y:-trans_y+warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
        extended_image[:img1.shape[0], :img1.shape[1]] = img1
        trans_y = 0

    return trans_x,trans_y,extended_image

def right_stitch_new(start,end,x_translation,y_translation):
    right_img=images[end].copy()
    i=end-1
    while(i>start-1):
        img1=images[i].copy()
        matched_points_img1,matched_points_img2=give_matched_points(i,x_translation,y_translation)
        adjusted_homography_matrix,calc_warp_w1,calc_warp_h1,max_y, min_y, trans_x, trans_y=trans_rotation_homography(matched_points_img2,matched_points_img1,right_img.shape[0],right_img.shape[1],img1.shape[0],img1.shape[1])
        x_translation,y_translation,right_img=new_stritched_image_right(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img1)
        plt.imshow(right_img)
        plt.show()
        i-=1
    return right_img,y_translation


def left_and_right(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img1):
    warp_image2 = warpPerspective(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1)
    extended_image = np.ones(((max(math.ceil(max_y),img1.shape[0])-min(math.floor(min_y),0)), warp_image2.shape[1] - trans_x, 3), dtype=np.uint8) * 255
    if min_y<=0:
    
        extended_image[:warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
        extended_image[trans_y:trans_y+img1.shape[0], :img1.shape[1]] = img1
    else:
    
        extended_image[-trans_y:-trans_y+warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
        extended_image[:img1.shape[0], :img1.shape[1]] = img1

    return extended_image

def stitch_right_left_new(p1,p2,left_img,right_img1,l_t_x,l_t_y,r_t_y):
    right_img=right_img1.copy()
    img1=left_img.copy()
    matched_points_img1,matched_points_img2=give_matched_points(p1,l_t_x,l_t_y)
    matched_points_img2[:,1]+=r_t_y
    adjusted_homography_matrix,calc_warp_w1,calc_warp_h1,max_y, min_y, trans_x, trans_y=trans_rotation_homography(matched_points_img2,matched_points_img1,right_img.shape[0],right_img.shape[1],img1.shape[0],img1.shape[1])
    return left_and_right(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1,trans_x, trans_y,max_y, min_y,img1)

def stitch(num_imgs):
    if(num_imgs%2==0):
        print('Left Side')
        idx_up = ((num_imgs)//2)-1
        idx_low = max(idx_up-2,0)
        left_img, l_t_x, l_t_y = left_stitch_new(idx_low, idx_up, 0, 0)
        print()
        print('Right Side')
        idx_low = (num_imgs)//2
        idx_up = min(idx_low+1,num_imgs-1)
        right_img, r_t_y = right_stitch_new(idx_low, idx_up, 0, 0)
        print()
        print('Final pic')
        complete = stitch_right_left_new(num_imgs//2-1, num_imgs//2, left_img, right_img, l_t_x, l_t_y, r_t_y)
    else:
        print('Left Side')
        idx_up = (num_imgs)//2
        idx_low = max(idx_up-2,0)
        left_img, l_t_x, l_t_y = left_stitch_new(idx_low, idx_up, 0, 0)
        print()
        print('Right Side')
        idx_low = ((num_imgs)//2)+1
        idx_up = min(idx_low+1,num_imgs-1)
        print(idx_low,idx_up)
        right_img, r_t_y = right_stitch_new(idx_low, idx_up, 0, 0)
        print('Final pic')
        complete = stitch_right_left_new(num_imgs//2, num_imgs//2+1, left_img, right_img, l_t_x, l_t_y, r_t_y)
    plt.imshow(complete)
    plt.show()
    return complete

final_img = stitch(len(images))
cv2.imwrite(sys.argv[3]+'/Panorama.jpg', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
print('End')
