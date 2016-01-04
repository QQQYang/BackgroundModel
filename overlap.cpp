#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "overlap.h"

using namespace cv;
using namespace std;

//typedef struct Point
//{
//	int x;
//	int y;
//}Point;

bool IsRectCross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
{
	bool ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
		min(q1.x, q2.x) <= max(p1.x, p2.x) &&
		min(p1.y, p2.y) <= max(q1.y, q2.y) &&
		min(q1.y, q2.y) <= max(p1.y, p2.y);
	//Rect r = Rect(p1, p2) & Rect(q1, q2);
	return ret;
}
//跨立判断
bool IsLineSegmentCross(const Point &pFirst1, const Point &pFirst2, const Point &pSecond1, const Point &pSecond2)
{
	if ((pFirst1 - pSecond1).cross(pSecond1 - pSecond2) * (pFirst2 - pSecond1).cross(pSecond1 - pSecond2) > 0)//不相交
		return false;
	if ((pSecond1 - pFirst1).cross(pFirst1 - pFirst2) * (pSecond2 - pFirst1).cross(pFirst1 - pFirst2) > 0) //不相交
		return false;
	if ((pFirst1 - pSecond1).cross(pSecond1 - pSecond2) * (pFirst2 - pSecond1).cross(pSecond1 - pSecond2) == 0)
	{
		if ((pFirst1 - pFirst2).cross(pSecond1 - pSecond2) != 0)
			return true;
		else
			return false;
	}
	if ((pSecond1 - pFirst1).cross(pFirst1 - pFirst2) * (pSecond2 - pFirst1).cross(pFirst1 - pFirst2) == 0)
	{
		if ((pFirst1 - pFirst2).cross(pSecond1 - pSecond2) != 0)
			return true;
		else
			return false;
	}
	return true;
	//long line1, line2;
	//line1 = pFirst1.x * (pSecond1.y - pFirst2.y) +
	//	pFirst2.x * (pFirst1.y - pSecond1.y) +
	//	pSecond1.x * (pFirst2.y - pFirst1.y);
	//line2 = pFirst1.x * (pSecond2.y - pFirst2.y) +
	//	pFirst2.x * (pFirst1.y - pSecond2.y) +
	//	pSecond2.x * (pFirst2.y - pFirst1.y);
	//if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
	//	return false;

	//line1 = pSecond1.x * (pFirst1.y - pSecond2.y) +
	//	pSecond2.x * (pSecond1.y - pFirst1.y) +
	//	pFirst1.x * (pSecond2.y - pSecond1.y);
	//line2 = pSecond1.x * (pFirst2.y - pSecond2.y) +
	//	pSecond2.x * (pSecond1.y - pFirst2.y) +
	//	pFirst2.x * (pSecond2.y - pSecond1.y);
	//if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
	//	return false;
	//return true;
}

bool GetCrossPoint(const Point &p1, const Point &p2, const Point &q1, const Point &q2, long &x, long &y)
{
	if (IsRectCross(p1, p2, q1, q2))
	{
		if (IsLineSegmentCross(p1, p2, q1, q2))
		{
			//求交点:有问题
			//if (p1.x == p2.x) //L1平行于y轴
			//{
			//	if (q1.x == q2.x) //L2平行于y轴

			//}
			long tmpLeft, tmpRight;
			tmpLeft = (q2.x - q1.x) * (p1.y - p2.y) - (p2.x - p1.x) * (q1.y - q2.y);
			tmpRight = (p1.y - q1.y) * (p2.x - p1.x) * (q2.x - q1.x) + q1.x * (q2.y - q1.y) * (p2.x - p1.x) - p1.x * (p2.y - p1.y) * (q2.x - q1.x);

			x = (int)((double)tmpRight / (double)tmpLeft);

			tmpLeft = (p1.x - p2.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q1.x - q2.x);
			tmpRight = p2.y * (p1.x - p2.x) * (q2.y - q1.y) + (q2.x - p2.x) * (q2.y - q1.y) * (p1.y - p2.y) - q2.y * (q1.x - q2.x) * (p2.y - p1.y);
			y = (int)((double)tmpRight / (double)tmpLeft);
			return true;
		}
	}
	return false;
}

//  The function will return YES if the point x,y is inside the polygon, or
//  NO if it is not.  If the point is exactly on the edge of the polygon,
//  then the function may return YES or NO.
bool IsPointInpolygon(std::vector<Point> poly, Point pt)
{
	int i, j;
	bool c = false;
	for (i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
	{
		if ((((poly[i].y <= pt.y) && (pt.y < poly[j].y)) ||
			((poly[j].y <= pt.y) && (pt.y < poly[i].y)))
			&& (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x))
		{
			c = !c;
		}
	}
	return c;
}


//若点a大于点b,即点a在点b顺时针方向,返回true,否则返回false
bool PointCmp(const Point &a, const Point &b, const Point &center)
{
	if (a.x >= 0 && b.x < 0)
		return true;
	if (a.x == 0 && b.x == 0)
		return a.y > b.y;
	//向量OA和向量OB的叉积
	int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
	if (det < 0)
		return true;
	if (det > 0)
		return false;
	//向量OA和向量OB共线，以距离判断大小
	int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
	int d2 = (b.x - center.x) * (b.x - center.y) + (b.y - center.y) * (b.y - center.y);
	return d1 > d2;
}
void ClockwiseSortPoints(std::vector<Point> &vPoints)
{
	//计算重心
	cv::Point center;
	double x = 0, y = 0;
	for (int i = 0; i < vPoints.size(); i++)
	{
		x += vPoints[i].x;
		y += vPoints[i].y;
	}
	center.x = (int)x / vPoints.size();
	center.y = (int)y / vPoints.size();

	//冒泡排序
	for (int i = 0; i < vPoints.size() - 1; i++)
	{
		for (int j = 0; j < vPoints.size() - i - 1; j++)
		{
			if (PointCmp(vPoints[j], vPoints[j + 1], center))
			{
				cv::Point tmp = vPoints[j];
				vPoints[j] = vPoints[j + 1];
				vPoints[j + 1] = tmp;
			}
		}
	}
}

bool PolygonClip(const vector<Point> &poly1, const vector<Point> &poly2, std::vector<Point> &interPoly)
{
	if (poly1.size() < 3 || poly2.size() < 3)
	{
		return false;
	}

	long x, y;
	//计算多边形交点
	for (int i = 0; i < poly1.size(); i++)
	{
		int poly1_next_idx = (i + 1) % poly1.size();
		for (int j = 0; j < poly2.size(); j++)
		{
			int poly2_next_idx = (j + 1) % poly2.size();
			if (GetCrossPoint(poly1[i], poly1[poly1_next_idx],
				poly2[j], poly2[poly2_next_idx],
				x, y))
			{
				if (find(interPoly.begin(), interPoly.end(), Point(x, y)) == interPoly.end())
					interPoly.push_back(cv::Point(x, y));
				else
					continue;
			}
		}
	}

	//计算多边形内部点
	for (int i = 0; i < poly1.size(); i++)
	{
		if (pointPolygonTest(poly2, poly1[i], false) > -1)
		{
			if (find(interPoly.begin(), interPoly.end(), poly1[i]) == interPoly.end())
				interPoly.push_back(poly1[i]);
			else
				continue;
		}
	}
	for (int i = 0; i < poly2.size(); i++)
	{
		if (pointPolygonTest(poly1, poly2[i], false) > -1)
		{
			if (find(interPoly.begin(), interPoly.end(), poly2[i]) == interPoly.end())
				interPoly.push_back(poly2[i]);
			else
				continue;
		}
	}

	if (interPoly.size() <= 0)
		return false;

	//点集排序 
	ClockwiseSortPoints(interPoly);
	return true;
}


bool ImageOverlap(size_t rows, size_t cols, Mat H, std::vector<cv::Point> &vPtsImg1, std::vector<cv::Point> &vPtsImg2)
{
	std::vector<cv::Point> vSrcPtsImg1;
	std::vector<cv::Point> vSrcPtsImg2;

	vSrcPtsImg1.push_back(cv::Point(0, 0));
	vSrcPtsImg1.push_back(cv::Point(0, rows));
	vSrcPtsImg1.push_back(cv::Point(cols, rows));
	vSrcPtsImg1.push_back(cv::Point(cols, 0));

	vSrcPtsImg2.push_back(cv::Point(0, 0));
	vSrcPtsImg2.push_back(cv::Point(0, rows));
	vSrcPtsImg2.push_back(cv::Point(cols, rows));
	vSrcPtsImg2.push_back(cv::Point(cols, 0));

	//计算图像2在图像1中对应坐标信息
	std::vector<cv::Point> vWarpPtsImg2;
	for (int i = 0; i < vSrcPtsImg2.size(); i++)
	{
		cv::Mat srcMat = Mat::zeros(3, 1, CV_64FC1);
		srcMat.at<double>(0, 0) = vSrcPtsImg2[i].x;
		srcMat.at<double>(1, 0) = vSrcPtsImg2[i].y;
		srcMat.at<double>(2, 0) = 1.0;

		cv::Mat warpMat = H * srcMat;
		cv::Point warpPt;
		warpPt.x = cvRound(warpMat.at<double>(0, 0) / warpMat.at<double>(2, 0));
		warpPt.y = cvRound(warpMat.at<double>(1, 0) / warpMat.at<double>(2, 0));

		vWarpPtsImg2.push_back(warpPt);
	}
	//计算图像1和转换后的图像2的交点
	if (!PolygonClip(vSrcPtsImg1, vWarpPtsImg2, vPtsImg1))
		return false;

	for (int i = 0; i < vPtsImg1.size(); i++)
	{
		cv::Mat srcMat = Mat::zeros(3, 1, CV_64FC1);
		srcMat.at<double>(0, 0) = vPtsImg1[i].x;
		srcMat.at<double>(1, 0) = vPtsImg1[i].y;
		srcMat.at<double>(2, 0) = 1.0;

		cv::Mat warpMat = H.inv() * srcMat;
		cv::Point warpPt;
		warpPt.x = cvRound(warpMat.at<double>(0, 0) / warpMat.at<double>(2, 0));
		warpPt.y = cvRound(warpMat.at<double>(1, 0) / warpMat.at<double>(2, 0));
		vPtsImg2.push_back(warpPt);
	}
	return true;
}