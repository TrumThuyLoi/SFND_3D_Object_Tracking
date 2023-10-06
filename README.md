# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * Install Git LFS before cloning this Repo.
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## How I Addressed Each Rubric Point

### FP.1 Match 3D Objects
    * I create a 2D matrix, size of first dimention is number of previous frame bounding boxes, size of the second dimention is number of current frame bounding boxes.
    * Then I loop through all matches and find the element of the matrix that the match keypoint belong to, then add 1 to that maxtrix element.
    * After the loop, I pic the highest number of keypoint correspondences for each previous frame bounding boxes and make it is the best pair bounding boxes.

### FP.2 Compute Lidar-based TTC
    * I remove 5% nearest lidar point of current and previous frame so we can void outliers.
    * After that, I find minXPrev (min x of previous frame) and minXCurr (min x of current frame)
    * Then, apply fomular: TTC = minXCurr * dT / (minXPrev - minXCurr)
    * with dT is the time between previous frame and current frame.

### FP.3 Associate Keypoint Correspondences with Bounding Boxes
    * I sorted kptMatches based on the euclidean distance between keypoints of each pairs in ascending order.
    * Then I remove 10% matched pairs (5% too close and 5% too far away) to avoid outliers. 
    * After taht, I store all the remain matches that have current frame keypoint inside ROI of current frame in the kptMatches attribute.

### FP.4 Compute Camera-based TTC
    * Firstly, I compute relative distances between keypoints in previous frame and current frame.
    * Note that we avoid any distPrev that nearly 0 and distCurr that smaller than minDist (in this case 100).
    * distPrev and distCurr then used to compute distRatios, and all distRatio is stored in a std::vector.
    * Then we find the median of distRatio.
    * Finaly, apply fomular: TTC = -dT / (1.0 - *itMedian);
    * with dT is the time between previous frame and current frame.

### FP.5 Performance Evaluation 1
    * I log the TTC of Lidar in ./dat/tracked_data folder. Some of them may are negative bacause in that case the preceding vehicle is move far away.
    * Because we are using constant velocity model so that they work does not correctly when relative velocity with preceding vehicle is decrease.

### FP.6 Performance Evaluation 2
    * There are some detector / descriptor combinations give some unexpected results.
    * The "nan" result because all distPrev and distCurr don't satisfy the conditions.
    * The "-inf" result because the ditsRatio median nearly 1.