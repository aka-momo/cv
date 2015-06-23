#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define X_AXIS 0
#define Y_AXIS 1

void grayImage(Mat& img, Mat& gray_image);
void opencvHarris(Mat &img);
void harris(Mat &img, Mat& harris, int block_size, float k);
void sobel(Mat &img, Mat& deriv, uint axis);
float gaussian(Point p, Point center, float sd);

