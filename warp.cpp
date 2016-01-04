#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <math.h>

using namespace cv;
using namespace std;

#define MAX_CORNER_COUNT 200
#define QUALITY_LEVEL 0.01
#define MIN_DIST 20.0
#define DESIRED_LABLE_COUNT 15

//输入为灰度图像
bool calH(Mat cur, Mat pre, Mat& H)
{
	if (cur.empty() || pre.empty())
	{
		return false;
	}
	vector<Point2f> preCorner, curCorner;

	//找角点
	vector<Point2f> features[2];
	vector<Point2f> motionForBlockKmeans;
	vector< vector<float> > motion;
	//	vector<Vec4f> motion;

	goodFeaturesToTrack(pre, features[0], MAX_CORNER_COUNT, QUALITY_LEVEL, MIN_DIST);
	//计算光流场
	if (!features[0].empty())
	{
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(pre, cur, features[0], features[1], status, err);
		//T.copyTo(last_T);
		//去掉一些不好的特征点
		vector<Point2f>::iterator featureCurIt = features[1].begin(), featurePreIt = features[0].begin();
		vector<uchar>::iterator statusIt = status.begin();
		for (; featureCurIt != features[1].end() && featurePreIt != features[0].end() && statusIt != status.end(); statusIt++)
		{
			if ((*statusIt) == 0 || (abs((*featureCurIt).x - (*featurePreIt).x) + abs((*featureCurIt).y - (*featurePreIt).y)) < 1)
			{
				if (featurePreIt == features[0].begin())
				{
					features[0].erase(featurePreIt);
					featurePreIt = features[0].begin();
				}
				else
				{
					featurePreIt = features[0].erase(featurePreIt);
				}

				if (featureCurIt == features[1].begin())
				{
					features[1].erase(featureCurIt);
					featureCurIt = features[1].begin();
				}
				else
				{
					featureCurIt = features[1].erase(featureCurIt);
				}

				if (featurePreIt == features[0].end() || featureCurIt == features[1].end())
					break;
			}
			else
			{
				vector<float> motionEle;
				//				vector<float> length(1), angle(1);

				preCorner.push_back(*featurePreIt);
				curCorner.push_back(*featureCurIt);
				//计算极角和极径
				float length = norm(*featureCurIt - *featurePreIt);
				float angle = atan2((*featureCurIt - *featurePreIt).y, (*featureCurIt - *featurePreIt).x) * 180 / M_PI;
				//				cartToPolar((*featureCurIt - *featurePreIt).x, (*featureCurIt - *featurePreIt).y, length, angle);
				motionEle.push_back(length);
				motionEle.push_back(angle);
				motionEle.push_back(((*featureCurIt).x - (*featurePreIt).x) > 0);
				//motionEle.push_back((*featureCurIt).x);//当前点的坐标
				//motionEle.push_back((*featureCurIt).y);
				//motionEle.push_back((*featureCurIt).x - (*featurePreIt).x);//当前点的运动矢量
				//motionEle.push_back((*featureCurIt).y - (*featurePreIt).y);
				//				motion.push_back(Vec4f((*featureCurIt).x, (*featureCurIt).y, (*featureCurIt).x - (*featurePreIt).x, (*featureCurIt).y - (*featurePreIt).y));
				motion.push_back(motionEle);
				motionForBlockKmeans.push_back(Point2f(length, angle));
				featureCurIt++;
				featurePreIt++;

			}
		}

		if (features[0].size() > 4)
		{
			H = findHomography(features[1], features[0], CV_LMEDS);
			return true;
		}
		else
			return false;
	}
	else
		return false;
}