#ifndef __OVERLAP_H__
#define __OVERLAP_H__

#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

bool ImageOverlap(size_t rows, size_t cols, Mat H, std::vector<cv::Point> &vPtsImg1, std::vector<cv::Point> &vPtsImg2);

#endif