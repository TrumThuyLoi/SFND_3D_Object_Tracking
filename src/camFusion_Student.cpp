
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for(cv::DMatch& match : kptMatches)
    {
        auto itPrevFrKpt = kptsPrev.begin() + match.queryIdx;
        auto itCurrFrKpt = kptsCurr.begin() + match.trainIdx;

        if(boundingBox.roi.contains(itPrevFrKpt->pt))
        {
            boundingBox.kptMatches.emplace_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios;
    double minDist = 100.0;
    for(auto itOuter = kptMatches.begin(); itOuter != kptMatches.end()-1; itOuter++)
    {
        auto itKpOuterPrev = kptsPrev.begin() + itOuter->queryIdx;
        auto itKpOuterCurr = kptsCurr.begin() + itOuter->trainIdx;

        for(auto itInner = kptMatches.begin()+1; itInner != kptMatches.end(); itInner++)
        {
            auto itKpInnerPrev = kptsPrev.begin() + itInner->queryIdx;
            auto itKpInnerCurr = kptsCurr.begin() + itInner->trainIdx;

            double distPrev = cv::norm(itKpOuterPrev->pt - itKpInnerPrev->pt);
            double distCurr = cv::norm(itKpOuterCurr->pt - itKpInnerCurr->pt);

            if(distPrev > std::numeric_limits<double>::epsilon() && distCurr > minDist)
            {   // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.emplace_back(distRatio);
            }
        }
    }

    if(distRatios.size() <= 0)
    {
        TTC = std::nan("distRatios.size() is 0");
        return;
    }

    auto itMedian = distRatios.begin() + distRatios.size()/2;
    std::nth_element(distRatios.begin(), itMedian, distRatios.end());
    double sumBeforeMedian = std::accumulate(distRatios.begin(), itMedian, 0);
    double sumAfterMedian = std::accumulate(itMedian, distRatios.end(), 0);
    std::cout << "sumBeforeMedian: " << sumBeforeMedian << " - sumAfterMedian: " << sumAfterMedian << "\n";
    assert(sumBeforeMedian < sumAfterMedian);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    auto lambdaFindMinX = [](std::vector<LidarPoint>& lidarPoints){
        double minX = 1e9;
        for(auto point : lidarPoints)
        {
            minX = minX > point.x ? point.x : minX;
        }
        return minX;
    };
    
    double minXPrev = lambdaFindMinX(lidarPointsPrev);
    double minXCurr = lambdaFindMinX(lidarPointsCurr);
    
    TTC = minXCurr * (1.0/frameRate) / (minXPrev - minXCurr); 
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::unordered_map<int, std::vector<int>> countMatches;
    for(BoundingBox& prevFrBox : prevFrame.boundingBoxes)
    {
        countMatches.insert({prevFrBox.boxID, std::vector<int>(currFrame.boundingBoxes.size(), 0)});
    }

    auto lambdaFindContainBBox = [](std::vector<BoundingBox>& bboxes, cv::KeyPoint& kp){
        int resultIdx = -1;
        for(BoundingBox& box : bboxes)
        {
            if(box.roi.contains(kp.pt) == false)
                continue;

            resultIdx = box.boxID;
            break;
        }
        return resultIdx;
    };

    for(cv::DMatch& match : matches)
    {
        // std::cout << "query descriptor index: " << match.queryIdx << "\ntrain descriptor index: " << match.trainIdx << "\n";
        auto itPrevFrPoint = prevFrame.keypoints.begin() + match.queryIdx;
        auto itCurrFrPoint = currFrame.keypoints.begin() + match.trainIdx;

        int prevFrBoxID = lambdaFindContainBBox(prevFrame.boundingBoxes, *itPrevFrPoint);
        int currFrBoxID = lambdaFindContainBBox(currFrame.boundingBoxes, *itCurrFrPoint);

        if(prevFrBoxID != -1 && currFrBoxID != -1)
        {
            countMatches[prevFrBoxID].at(currFrBoxID) += 1;
        }
    }

    for(BoundingBox& prevFrBox : prevFrame.boundingBoxes)
    {
        auto listCount = countMatches.find(prevFrBox.boxID);
        int maxIdx = max_element(listCount->second.begin(), listCount->second.end()) - listCount->second.begin();
        if(listCount->second[maxIdx] <= 0)
            continue;
        
        std::cout   << "Best match BBox prevFrBoxID: " << prevFrBox.boxID << ", classID: " << prevFrBox.classID 
                    << "; currFrBoxID: " << maxIdx << ", classID: " << currFrame.boundingBoxes[maxIdx].classID << "\n";

        bbBestMatches.insert({prevFrBox.boxID, maxIdx});
    }
}
