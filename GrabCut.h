#pragma once
#include <opencv/cv.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "graph.h"
using namespace cv;
using namespace std;
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );
    
	~GrabCut2D(void);
};

enum{
    PRINT_WEIGHT,
    PRINT_MEAN,
    PRINT_COV,
    PRINT_INVCOV,
    PRINT_DET,
    PRINT_ALL
};

class GMM{
public:
    static const int componentSize = 5;
    GMM(Mat& _model);
    double posibility(Vec3f color);
    double posibility(Vec3f color,int cid);
    void init();
    void learn();
    void addsample(Vec3f color, int which);
    void addsample(Vec3f color);
    int whichComponent(Vec3f color);
    void printinfo(int printtype);
    
private:
    double* weight;
    double* mean; //sample mean
    double* cov; //sample covariance
    Vec3f meancolor[componentSize];
    vector<Mat> covmat;
    vector<Mat> invcovmat;
    double covdet[componentSize];
    vector<vector<Vec3f>> samples;
};