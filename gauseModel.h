#ifndef __GAUSSMODEL_H__
#define __GAUSSMODEL_H__

#include <opencv2/core.hpp>
using namespace cv;

class GaussModel
{
public :
	Mat gm;	//存放高斯模型
	Mat gmCandidate; //存放候选高斯模型
	Mat gray; //存放高斯模型的灰度图
private:
	Mat mask;	//age
	const float variance = 400.f;
	const float thetaS = 2.f;
	const float thetaV = 50.f * 50;
	const float thetaD = 4.f;
	const uchar N = 4;
	const float lambda = 0.001;
public:
	bool initial(Mat src);
	void updateModel(Mat cur, Mat H, Mat& dst);
	Mat out();
	void drawAlpha(Mat& bk, size_t t);
	float avrNN(Mat src, Point2f p, uchar N);
	float varNN(Mat src, Point2f p, uchar N, bool flag);
private:
	void updateMask(Mat H); //更新age
	void updateGray(); //更新灰度图
};

#endif