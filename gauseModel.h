#ifndef __GAUSSMODEL_H__
#define __GAUSSMODEL_H__

#include <opencv2/core.hpp>
using namespace cv;

class GaussModel
{
public :
	Mat gm;	//存放高斯模型
	Mat gray; //存放高斯模型的灰度图
private:
	Mat mask;	//age
	const float variance = 400.f;
public:
	bool initial(Mat src);
	void updateModel(Mat cur, Mat H, Mat& dst);
	Mat out();
	void drawAlpha(Mat& bk, size_t t);
private:
	void updateMask(Mat H); //更新age
	void updateGray(); //更新灰度图
};

#endif