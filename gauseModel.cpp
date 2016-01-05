#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "gauseModel.h"
#include "overlap.h"

/*
2015/12/24调试笔记：
q1. 论文中对D的计算有问题：方差在分母上，则会导致方差越小，D越大；而实际情况是方差越小，说明改点更可能是背景
a1: 问作者
q2. 用背景模型和当前帧去计算单应矩阵时，不稳定；而用前一帧和当前帧去计算单应矩阵时，学习到的背景和实际的背景总是存在一定的错位
q3. 更新速度太慢
q4. mask总是显示不出来
*/


//作调试用
extern Mat lap;

using namespace cv;
using namespace std;

//计算N*N区域的平均值
float GaussModel::avrNN(Mat src, Point2f p, uchar N)
{
	float avr = 0.f;
	for (size_t i = p.y * N; i < p.y * N + N; i++)
	{
		uchar* srcData = src.ptr<uchar>(i);
		for (size_t j = p.x * N; j < p.x * N + N; j++)
		{
			avr += srcData[j];
		}
	}
	avr /= (N*N);
	return avr;
}

//计算N*N区域的方差
float GaussModel::varNN(Mat src, Point2f p, uchar N, bool flag)
{
	float var = 0.f;
	if (flag == true)
	{
		for (size_t i = p.y * N; i < p.y * N + N; i++)
		{
			uchar* srcData = src.ptr<uchar>(i);
			for (size_t j = p.x * N; j < p.x * N + N; j++)
			{
				float tempVar = (srcData[j] - gm.at<Vec3f>(p.y, p.x).val[0]) * (srcData[j] - gm.at<Vec3f>(p.y, p.x).val[0]);
				if (tempVar > var)
					var = tempVar;
			}
		}
	}
	else
	{
		for (size_t i = p.y * N; i < p.y * N + N; i++)
		{
			uchar* srcData = src.ptr<uchar>(i);
			for (size_t j = p.x * N; j < p.x * N + N; j++)
			{
				float tempVar = (srcData[j] - gmCandidate.at<Vec3f>(p.y, p.x).val[0]) * (srcData[j] - gmCandidate.at<Vec3f>(p.y, p.x).val[0]);
				if (tempVar > var)
					var = tempVar;
			}
		}
	}
	return var;
}

//判断一个区域的前景点与背景点


//三个参数的顺序:均值， 方差， 更新速率
//input: gray
bool GaussModel::initial(Mat src)
{
	gm = Mat::zeros(src.rows / N, src.cols / N, CV_32FC3);
	gmCandidate = Mat::zeros(src.rows / N, src.cols / N, CV_32FC3);
	src.copyTo(gray);
	if (src.empty())
		return false;
	else
	{
		for (size_t i = 0; i < gm.rows; i++)
		{
			Vec3f* gmData = gm.ptr<Vec3f>(i);
			Vec3f* gmCandidateData = gmCandidate.ptr<Vec3f>(i);
			for (size_t j = 0; j < gm.cols; j++)
			{
				float avr = avrNN(src, Point2f(j, i), N);
				gmData[j].val[0] = avr;
				gmData[j].val[1] = variance;
				gmData[j].val[2] = 1.f;
				gmCandidateData[j].val[0] = avr;
				gmCandidateData[j].val[1] = variance;
				gmCandidateData[j].val[2] = 1.f;
			}
		}
		mask = Mat::ones(src.size(), CV_16UC1);
		return true;
	}
}

void GaussModel::updateModel(Mat cur, Mat H, Mat& dst)
{
	//Mat warpFrame;
	//warpAffine(gray, warpFrame, H, gray.size(), INTER_LINEAR);
	//Mat mask(cur.size(), CV_8UC1);	//掩码
	//updateMask(H);
	dst = Mat::zeros(cur.size(), CV_8UC1);
	//Mat iH;
	//iH = H.inv(); //求逆

	/*****运动补偿*******/
	Mat gmCopy; //备份
	gm.copyTo(gmCopy);
	for (size_t i = 0; i < gm.rows; i++)
	{
		for (size_t j = 0; j < gm.cols; j++)
		{
			Point2f bkReal;
			Mat srcMat = Mat::zeros(3, 1, CV_64FC1);
			srcMat.at<double>(0, 0) = j*N + N / 2;
			srcMat.at<double>(0, 0) = i*N + N / 2;
			srcMat.at<double>(2, 0) = 1.0;
			Mat warpMat = H * srcMat;
			bkReal.x = warpMat.at<double>(0, 0) / warpMat.at<double>(2, 0);
			bkReal.y = warpMat.at<double>(1, 0) / warpMat.at<double>(2, 0);

			if ((bkReal.x - N / 2) >= 0 && (bkReal.x - N / 2) < gm.cols - 1 && (bkReal.y - N / 2) >= 0 && (bkReal.y - N / 2) < gm.rows - 1)
			{
				float wk[4] = { 0.f }; //权向量
				Rect2f neighbor[5];
				neighbor[0] = Rect2f((bkReal.x - N / 2), (bkReal.y - N / 2), N, N);
				neighbor[1] = Rect2f(floor((bkReal.x - N / 2) / (float)N) * N, floor((bkReal.y - N / 2) / (float)N) * N, N, N);
				neighbor[2] = Rect2f(floor((bkReal.x - N / 2) / (float)N) * N, ceilf((bkReal.y - N / 2) / (float)N) * N, N, N);
				neighbor[3] = Rect2f(ceilf((bkReal.x - N / 2) / (float)N) * N, floor((bkReal.y - N / 2) / (float)N) * N, N, N);
				neighbor[4] = Rect2f(ceilf((bkReal.x - N / 2) / (float)N) * N, ceilf((bkReal.y - N / 2) / (float)N) * N, N, N);

				wk[0] = (neighbor[1] & neighbor[0]).area() / (N*N);
				wk[1] = (neighbor[2] & neighbor[0]).area() / (N*N);
				wk[2] = (neighbor[3] & neighbor[0]).area() / (N*N);
				wk[3] = 1 - wk[0] - wk[1] - wk[2];
				gm.at<Vec3f>(i, j).val[0] = wk[0] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0]
					+ wk[1] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0]
					+ wk[2] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0]
					+ wk[3] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0];

				gm.at<Vec3f>(i, j).val[1] = wk[0] * (gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[1] + gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0] - gm.at<Vec3f>(i, j).val[0] * gm.at<Vec3f>(i, j).val[0])
					+ wk[1] * (gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[1] + gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0] - gm.at<Vec3f>(i, j).val[0] * gm.at<Vec3f>(i, j).val[0])
					+ wk[2] * (gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[1] + gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[0] - gm.at<Vec3f>(i, j).val[0] * gm.at<Vec3f>(i, j).val[0])
					+ wk[3] * (gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[1] + gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[0] - gm.at<Vec3f>(i, j).val[0] * gm.at<Vec3f>(i, j).val[0]);

				gm.at<Vec3f>(i, j).val[2] = wk[0] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[2]
					+ wk[1] * gmCopy.at<Vec3f>(floor((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[2]
					+ wk[2] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), floor((bkReal.x - N / 2) / (float)N)).val[2]
					+ wk[3] * gmCopy.at<Vec3f>(ceilf((bkReal.y - N / 2) / (float)N), ceilf((bkReal.x - N / 2) / (float)N)).val[2];
			}

			if (gm.at<Vec3f>(i, j).val[1] > thetaV)
				gm.at<Vec3f>(i, j).val[2] = gm.at<Vec3f>(i, j).val[2] * exp((-1) * lambda * (gm.at<Vec3f>(i, j).val[1] - thetaV));
		}
	}

	for (size_t i = 0; i < gm.rows; i++)
	{
		for (size_t j = 0; j < gm.cols; j++)
		{
			Point2f bkReal(j, i);
			float avr = avrNN(cur, bkReal, N);
			float alpha = gm.at<Vec3f>(bkReal.y, bkReal.x).val[2];

			/***************更新模型****************/

			if ((avr - gm.at<Vec3f>(bkReal.y, bkReal.x).val[0]) * (avr - gm.at<Vec3f>(bkReal.y, bkReal.x).val[0]) < thetaS * gm.at<Vec3f>(bkReal.y, bkReal.x).val[1]) //更新apparent
			{
				gm.at<Vec3f>(bkReal.y, bkReal.x).val[0] = alpha / (1 + alpha) * gm.at<Vec3f>(bkReal.y, bkReal.x).val[0] + 1 / (1 + alpha) * avr;
				float var = varNN(cur, bkReal, N, true);
				gm.at<Vec3f>(bkReal.y, bkReal.x).val[1] = alpha / (1 + alpha) * gm.at<Vec3f>(bkReal.y, bkReal.x).val[1] + 1 / (1 + alpha) * var;
				gm.at<Vec3f>(bkReal.y, bkReal.x).val[2] ++;
			}
			else if ((avr - gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[0]) * (avr - gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[0]) < thetaS * gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[1]) // 更新candidate
			{
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[0] = alpha / (1 + alpha) * gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[0] + 1 / (1 + alpha) * avr;
				float var = varNN(cur, bkReal, N, false);
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[1] = alpha / (1 + alpha) * gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[1] + 1 / (1 + alpha) * var;
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[2] ++;
			}
			else //初始化candidate
			{
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[0] = avr;
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[1] = variance;
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[2] = 1.f;
			}

			if (gmCandidate.at<Vec3f>(bkReal.y, bkReal.x).val[2] > gm.at<Vec3f>(bkReal.y, bkReal.x).val[2]) //交换apparent 和 candidate
			{
				Vec3f temp = gmCandidate.at<Vec3f>(bkReal.y, bkReal.x);
				gmCandidate.at<Vec3f>(bkReal.y, bkReal.x) = gm.at<Vec3f>(bkReal.y, bkReal.x);
				gm.at<Vec3f>(bkReal.y, bkReal.x) = temp;
			}

			/**************判断前景点和背景点***************/
			for (size_t m = i*N; m<(i + 1)*N; m++)
			{
				uchar* curData = cur.ptr<uchar>(m);
				uchar* dstData = dst.ptr<uchar>(m);
				for (size_t n = j*N; n<(j + 1)*N; n++)
				{
					if ((curData[n] - gm.at<Vec3f>(bkReal.y, bkReal.x).val[0]) * (curData[n] - gm.at<Vec3f>(bkReal.y, bkReal.x).val[0]) > thetaD * gm.at<Vec3f>(bkReal.y, bkReal.x).val[1])
						dstData[n] = 255;
					else
						dstData[n] = 0;
				}
			}
		}
	}
}

void GaussModel::updateMask(Mat H)
{
	updateGray();
	vector<Point> vPtsImg1, vPtsImg2;
	if (ImageOverlap(gm.rows, gm.cols, H, vPtsImg1, vPtsImg2)) //有重叠部分
	{
		RotatedRect minRect = minAreaRect(vPtsImg1);
		Point2f vertices[4];
		vector<Point2f> vPts;
		minRect.points(vertices);
		for (size_t i = 0; i < 4; i++)
		{
			line(lap, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 1, 8);
			vPts.push_back(vertices[i]);
		}

		for (size_t i = 0; i < mask.rows; i++)
		{
			ushort* maskData = mask.ptr<ushort>(i);
			for (size_t j = 0; j < mask.cols; j++)
			{
				if (pointPolygonTest(vPts, Point2f(j, i), false) > -1)	//在重叠区域内部
				{
					maskData[j] ++;
				}
				else  //不在重叠区域
				{
					maskData[j] = 1;
				}
			}
		}
	}
	//Mat maskForShow;
	//normalize(mask, maskForShow, 0, 255, NORM_MINMAX);
	//cout << mask.at<ushort>(480-1, 0) << endl;
	//cout << gm.at<Vec3f>(480 - 1, 0).val[0] << endl;
	//imshow("mask", maskForShow);
}

void GaussModel::updateGray()
{
	for (size_t i = 0; i < gm.rows; i++)
	{
		Vec3f* gmData = gm.ptr<Vec3f>(i);
		uchar* grayData = gray.ptr<uchar>(i);
		for (size_t j = 0; j < gm.cols; j++)
		{
			//类型转换
			grayData[j] = gmData[j].val[0];
		}
	}
}

//绘制alpha变化曲线
void GaussModel::drawAlpha(Mat& bk, size_t t)
{
	if (t < 1000)
	{
		
	}
	else
	{

	}
}
