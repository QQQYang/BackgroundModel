#include "pointsMatch.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

#define RATIO_TEST

//ORB特征点匹配
void pointsMatchOrb(Mat preFrame, Mat curFrame, vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch>& goodMatches)
{
	Ptr<ORB> detector = ORB::create(10000);
	vector<KeyPoint> keyPoints1, keyPoints2;
	detector->detect(preFrame, keyPoints1);
	detector->detect(curFrame, keyPoints2);
	Mat desc1, desc2;
	detector->compute(preFrame, keyPoints1, desc1);
	detector->compute(curFrame, keyPoints2, desc2);
#ifdef RATIO_TEST
	BFMatcher matcher;
	vector< vector<DMatch> > matches;
	matcher.knnMatch(desc1, desc2, matches, 2);

	//p1.assign(keyPoints1.begin(), keyPoints1.end());
	//p2.assign(keyPoints2.begin(), keyPoints2.end());
	vector<Point2f> pp1, pp2;
	for (uint i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < matches[i][1].distance * 0.4)
		{
			pp1.push_back((keyPoints1[matches[i][0].queryIdx]).pt);
			pp2.push_back((keyPoints2[matches[i][0].trainIdx]).pt);
		}
	}
	if (!pp1.empty())
	{
		vector<unsigned char> inliersMask(p1.size());
		findHomography(pp2, pp1, CV_FM_RANSAC, 3, inliersMask);
		for (uint i = 0; i < inliersMask.size(); i++)
		{
			if (inliersMask[i])
			{
				p1.push_back((keyPoints1[matches[i][0].queryIdx]));
				p2.push_back((keyPoints2[matches[i][0].trainIdx]));
				goodMatches.push_back(matches[i][0]);
			}
		}
	}
#else
	BFMatcher matcher(NORM_L2,true);
	vector<DMatch> matches;
	matcher.match(desc1, desc2, matches);

	//p1.assign(keyPoints1.begin(), keyPoints1.end());
	//p2.assign(keyPoints2.begin(), keyPoints2.end());
	vector<Point2f> pp1, pp2;
	for (uint i = 0; i < matches.size(); i++)
	{
		pp1.push_back((keyPoints1[matches[i].queryIdx]).pt);
		pp2.push_back((keyPoints2[matches[i].trainIdx]).pt);
	}
	if (!pp1.empty())
	{
		vector<unsigned char> inliersMask(p1.size());
		findHomography(pp2, pp1, CV_FM_RANSAC, 3, inliersMask);
		for (uint i = 0; i < inliersMask.size(); i++)
		{
			if (inliersMask[i])
			{
				p1.push_back((keyPoints1[matches[i].queryIdx]));
				p2.push_back((keyPoints2[matches[i].trainIdx]));
				goodMatches.push_back(matches[i]);
			}
		}
	}
#endif
}

void pointsMatch::showKeyPoints(Mat& dst)
{
	if (!p2.empty())
		drawKeypoints(curFrame, p2, dst, Scalar(0, 255, 0));
}

void pointsMatch::showMatches(Mat& dst)
{
	if (!goodMatches.empty())
		drawMatches(preFrame, p1, curFrame, p2, goodMatches, dst);
}

vector<Point2f> pointsMatch::getPoints(vector<KeyPoint> p)
{
	vector<Point2f> points;
	for (size_t i = 0; i < p.size(); i++)
	{
		points.push_back(p[i].pt);
	}
	return points;
}

bool pointsMatch::getKeyPoints()
{
	Ptr<ORB> detector = ORB::create(10000);
	vector<KeyPoint> keyPoints1, keyPoints2;
	detector->detect(preFrame, keyPoints1);
	detector->detect(curFrame, keyPoints2);
	Mat desc1, desc2;
	detector->compute(preFrame, keyPoints1, desc1);
	detector->compute(curFrame, keyPoints2, desc2);
	if (isRatioTest)	//比率测试 
	{
		BFMatcher matcher;
		vector< vector<DMatch> > matches;
		matcher.knnMatch(desc1, desc2, matches, 2);
		vector<Point2f> pp1, pp2;
		for (size_t i = 0; i < matches.size(); i++)
		{
			if (matches[i][0].distance < matches[i][1].distance * 0.4)
			{
				pp1.push_back((keyPoints1[matches[i][0].queryIdx]).pt);
				pp2.push_back((keyPoints2[matches[i][0].trainIdx]).pt);
			}
		}
		if (!pp1.empty())
		{
			vector<unsigned char> inliersMask(p1.size());
			findHomography(pp2, pp1, CV_FM_RANSAC, 3, inliersMask);
			for (size_t i = 0; i < inliersMask.size(); i++)
			{
				if (inliersMask[i])
				{
					p1.push_back((keyPoints1[matches[i][0].queryIdx]));
					p2.push_back((keyPoints2[matches[i][0].trainIdx]));
					goodMatches.push_back(matches[i][0]);
				}
			}
			if (p1.size() > 4)
				return true;
			else
				return false;
		}
		else
			return false;
	}
	else //交叉过滤
	{
		BFMatcher matcher(NORM_L2, true);
		vector<DMatch> matches;
		matcher.match(desc1, desc2, matches);

		//p1.assign(keyPoints1.begin(), keyPoints1.end());
		//p2.assign(keyPoints2.begin(), keyPoints2.end());
		vector<Point2f> pp1, pp2;
		for (uint i = 0; i < matches.size(); i++)
		{
			pp1.push_back((keyPoints1[matches[i].queryIdx]).pt);
			pp2.push_back((keyPoints2[matches[i].trainIdx]).pt);
		}
		if (!pp1.empty())
		{
			vector<unsigned char> inliersMask(p1.size());
			findHomography(pp2, pp1, CV_FM_RANSAC, 3, inliersMask);
			for (uint i = 0; i < inliersMask.size(); i++)
			{
				if (inliersMask[i])
				{
					p1.push_back((keyPoints1[matches[i].queryIdx]));
					p2.push_back((keyPoints2[matches[i].trainIdx]));
					goodMatches.push_back(matches[i]);
				}
			}
			if (p1.size() > 4)
				return true;
			else
				return false;
		}
		else
			return false;
	}
}

//构造函数不能指定默认值
pointsMatch::pointsMatch(Mat pre, Mat cur, bool matchMethod)
{
	pre.copyTo(preFrame);
	cur.copyTo(curFrame);
	isRatioTest = matchMethod;
}