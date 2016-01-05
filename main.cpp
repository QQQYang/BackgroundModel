#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include "gauseModel.h"
#include "warp.h"
#include "pointsMatch.h"

using namespace std;
using namespace cv;

Mat lap;

#define CAMERA

int main()
{
	VideoCapture cap;
	Mat curFrame, preFrame, nextFrame;
	Mat dst;
	int framePosition = 0;
	uint curFrameCount = 0;
	bool stop = false;
	bool isInitial = false;

	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setVarThreshold(10);

	GaussModel gModel;

#ifdef CAMERA
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "打开摄像头失败！" << endl;
		return -1;
	}
#else
	cap.open("r.avi");
	if (!cap.isOpened())
	{
		cout << "打开视频文件失败" << endl;
		return -1;
	}
	uint totalFrame = cap.get(CAP_PROP_FRAME_COUNT);

	namedWindow("src");
	createTrackbar("frameCurPosition", "src", &framePosition, totalFrame);
#endif

	Mat last_T;
	while (1)
	{
		if (!cap.read(curFrame))
		{
			cout << "读取视频失败" << endl;
			return -1;
		}
#ifdef CAMERA
		if (preFrame.empty())
			curFrame.copyTo(preFrame);
		else
		{
			curFrame.copyTo(lap);
			Mat curGray, preGray;
			cvtColor(curFrame, curGray, CV_BGR2GRAY);
			cvtColor(preFrame, preGray, CV_BGR2GRAY);
			if (!isInitial)
			{
				if (gModel.initial(curGray))
					isInitial = true;
			}
			else
			{
				Mat H;
				//pointsMatch p(preGray, curGray, true);
				//if (p.getKeyPoints())
				//{
				//	H = findHomography(p.getPoints(p.p2), p.getPoints(p.p1), CV_LMEDS);
				//	if (!isInitial)	//初始化
				//	{
				//		if (gModel.initial(curGray))
				//			isInitial = true;
				//	}
				//	else  //更新
				//	{
				//		gModel.updateModel(curGray, H, dst);
				//		//p.showKeyPoints(dst);
				//	}
				//}
				if (calH(curGray, preGray, H))
				{
					if (!isInitial)	//初始化
					{
						if (gModel.initial(curGray))
							isInitial = true;
					}
					else  //更新
					{
						gModel.updateModel(curGray, H, dst);
						
					}
				}
			}
			
			if (!dst.empty())
			{
				imshow("result", dst);
			}
			//imshow("overlap", lap);
			//imshow("gray", gModel.gray);
		}
		swap(preFrame, curFrame);
#else
		if (stop)
		{
			cap.set(CAP_PROP_POS_FRAMES, framePosition);
			curFrameCount = framePosition;
		}
		/****图像处理部分*****/
		if (curFrameCount > 0 && curFrameCount < totalFrame)
		{
			cap.get(CAP_PROP_POS_FRAMES);//获取当前帧号
			cap.set(CAP_PROP_POS_FRAMES, curFrameCount - 1);
			cap >> preFrame;
			cap.set(CAP_PROP_POS_FRAMES, curFrameCount + 1);
			cap >> nextFrame;
			cap.set(CAP_PROP_POS_FRAMES, curFrameCount);
			cap >> curFrame;

			if (preFrame.data)
			{
				//光流法
				//myOptiocalFlowCal(curFrame, preFrame, dst);

				//全局运动估计
				Point2d shift, shiftXY;
				Mat calibCur;
				phaseCorrelation(preFrame, curFrame, shift, shiftXY, calibCur);
				cout << "S:" << shift.x << setprecision(3) << "  " << "R:" << shift.y << setprecision(3) << endl;
				cout << "X:" << shiftXY.x << setprecision(3) << "  " << "Y:" << shiftXY.y << setprecision(3) << endl;
				//cout << "y:" << shift.y << endl;

				myOptiocalFlowCal(curFrame, calibCur, dst);
				imshow("result", dst);
			}

		}
#endif

		//imshow("src", curFrame);
		//		swap(preFrame, curFrame);
		curFrameCount++;
		/****图像处理部分*****/
		char button = waitKey(1);

#ifdef CAMERA
		if (button == 113)
			break;
#else
		if (button == 113 || cap.get(CAP_PROP_POS_FRAMES) == totalFrame) //按下q则退出

			break;

		else if (button == 32)//按下空格则暂停

		{

			stop = 1 - stop;

		}
#endif

	}
	return 0;
}