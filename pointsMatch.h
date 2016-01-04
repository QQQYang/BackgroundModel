#ifndef __POINTSMATCH_H
#define __POINTSMATCH_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;
//ORB特征点匹配
void pointsMatchOrb(Mat preFrame, Mat curFrame, vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch>& goodMatches);


class pointsMatch
{
public:
	Mat preFrame;
	Mat curFrame;
	bool isRatioTest = true;
	vector<KeyPoint> p1;
	vector<KeyPoint> p2;
	vector<DMatch> goodMatches;
public:
	void showMatches(Mat& dst);
	void showKeyPoints(Mat& dst);
	vector<Point2f> getPoints(vector<KeyPoint> p);
	bool getKeyPoints();
	//构造函数
	pointsMatch(Mat pre, Mat cur, bool matchMethod);
};

#endif