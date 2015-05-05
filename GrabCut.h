#pragma once
#include <opencv/cv.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "gcgraph.hpp"
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

class GaussianMixtureModel{
public:
    const static int sizeofGMM = 5;
    GaussianMixtureModel(Mat& _params);
    double probability(const Vec3b BGR);
    double probabilityInComponent(int component,const Vec3b BGR);
    vector<vector<Vec3b>> samples;
    void learning();
    int maxComponent(const Vec3b BGR);
    void addSample(Vec3b BGR);
private:
    // parameters for each gaussian model.
    void calcDetandInv();
    double mean[sizeofGMM][3]; //BGR 3-channel model
    double weight[sizeofGMM]; //weight for each component.
    double cov[sizeofGMM][3][3]; // cov matrix for each component.
    int totalsize() const;
    double covDeterminant[sizeofGMM];
    Mat covInv[sizeofGMM];
};