
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout << matcherType << " matching completed.\n";
    }

    double time_it_takes = 0.0;
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time_it_takes = 1000 * t / 1.0;
        std::cout << "(NN) with n=" << matches.size() << " matches in " << time_it_takes << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<vector<cv::DMatch>> knn_matches;
        t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time_it_takes = 1000 * t / 1.0;
        std::cout << "(KNN) with n=" << knn_matches.size() << " matches in " << time_it_takes << " ms" << endl;

        // filter matches using descriptor distance ratio test
        const double ratioThreshold = 0.8;
        for (std::vector<cv::DMatch>& knn_match : knn_matches)
        {
            if (knn_match[0].distance < ratioThreshold * knn_match[1].distance)
            {
                matches.emplace_back(knn_match[0]);
            }
        }
        std::cout << "Distance ratio test removed " << knn_matches.size() - matches.size() << " keypoints"<< endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        int                 nfeatures = 500;
        float               scaleFactor = 1.2f;
        int                 nlevels = 8;
        int                 edgeThreshold = 31;
        int                 firstLevel = 0;
        int                 WTA_K = 2;
        cv::ORB::ScoreType  scoreType = cv::ORB::HARRIS_SCORE;
        int                 patchSize = 31;
        int                 fastThreshold = 20; 

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        bool                orientationNormalized = true;
        bool                scaleNormalized = true;
        float               patternScale = 22.0f;
        int                 nOctaves = 4;
        std::vector<int>    selectedPairs;

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType   descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int                         descriptor_size = 0;
        int                         descriptor_channels = 3;
        float                       threshold = 0.001f;
        int                         nOctaves = 4;
        int                         nOctaveLayers = 4;
        cv::KAZE::DiffusivityType   diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        int     nfeatures = 0;
        int     nOctaveLayers = 3;
        double  contrastThreshold = 0.04;
        double  edgeThreshold = 10.0;
        double  sigma = 1.6;

        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else 
    {
        std::cout << "FEATURE DESCRIPTION failed. Wrong descriptorType - " << descriptorType << ". Use one of the following descriptors: BRISK, ORB, FREAK, AKAZE, SIFT" << endl;
        exit(-1);
    }

    // perform feature description
    std::cout << descriptorType << " Perform feature description\n";
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double time_it_takes = 1000 * t / 1.0;
    std::cout << descriptorType << " descriptor extraction in " << time_it_takes << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;
    int apertureSize = 3;
    float k = 0.04;

    double t = (double)cv::getTickCount();
    cv::Mat harris_img = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, harris_img, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    cv::Mat harris_norm_img, harris_norm_scale_img;
    cv::normalize(harris_img, harris_norm_img, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(harris_norm_img, harris_norm_scale_img);

    int R_threshold = 100;
    for(int r=0; r < harris_norm_scale_img.rows; r++)
    {
        for(int c=0; c < harris_norm_scale_img.cols; c++)
        {
            int response = harris_norm_scale_img.at<uint8_t>(r, c);
            if(response > R_threshold)
            {
                keypoints.emplace_back(c, r, 2*apertureSize);
                keypoints.back().response = response;
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double time_it_takes = 1000*t / 1.0;
    std::cout << "HARRIS detection with n= " << keypoints.size() << " keypoints in " << time_it_takes << "ms\n";

    if(bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    if(detectorType.compare("FAST") == 0)
    {
        detector = cv::FastFeatureDetector::create(10, true, cv::FastFeatureDetector::TYPE_9_16);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        int     threshold = 30;         // FAST/AGAST detection threshold score.
        int     octaves = 3;            // detection octaves (use 0 to do single scale)
        float   patternScale = 1.0f;    // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        detector = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(detectorType.compare("ORB") == 0)
    {
        int                 nfeatures = 500;
        float               scaleFactor = 1.2f;
        int                 nlevels = 8;
        int                 edgeThreshold = 31;
        int                 firstLevel = 0;
        int                 WTA_K = 2;
        cv::ORB::ScoreType  scoreType = cv::ORB::HARRIS_SCORE;
        int                 patchSize = 31;
        int                 fastThreshold = 20; 

        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType   descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int                         descriptor_size = 0;
        int                         descriptor_channels = 3;
        float                       threshold = 0.001f;
        int                         nOctaves = 4;
        int                         nOctaveLayers = 4;
        cv::KAZE::DiffusivityType   diffusivity = cv::KAZE::DIFF_PM_G2;

        detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        int     nfeatures = 0;
        int     nOctaveLayers = 3;
        double  contrastThreshold = 0.04;
        double  edgeThreshold = 10.0;
        double  sigma = 1.6;

        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else 
    {
        std::cout << "DETECT Keypoints failed. Wrong detectorType - " << detectorType << ". Use one of the following detector: FAST, BRIEF, ORB, FREAK, AKAZE, SIFT" << endl;
        exit(-1);
    }

    std::cout << "run detKeypointsModern with " << detectorType << "\n";
    double t = cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double time_it_takes = 1000*t / 1.0;
    std::cout << detectorType << " detection with n= " << keypoints.size() << " keypoints in " << time_it_takes << "ms\n";

    if(bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoints detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}