//
//  BorderMattingHandler.h
//  myGrabcut
//
//  Created by DarkTango on 6/2/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __myGrabcut__BorderMattingHandler__
#define __myGrabcut__BorderMattingHandler__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <random>
using namespace cv;
using namespace std;
struct Line{
    Point src;
    Point dst;
    double dist(Point2d p,bool testsign = false);
    bool isInside(Point2d p);
};

enum{
    BM_B=0, //background
    BM_U=100, //undefined
    BM_F=255 //foreground
};

//gaussian distribution
class Gauss{
public:
    Gauss();
    static double gauss(const double center, const double width, const double r);
    static double probability(const Vec3f& mean, const Mat& covmat, Vec3f color);
    void addsample(Vec3f color);
    void learn();
    Vec3f getmean()const{return mean;}
    Mat getcovmat()const{return covmat;}
private:
    Vec3f mean;
    Mat covmat;
    vector<Vec3f> samples;
};

class BorderMattingHandler{
public:
    BorderMattingHandler(const Mat& image, vector<Point>& contour);
    vector<Line> contours;
    int minestDistanceContourIndex(Point p);
    void constructTrimap();
    double dataTermForPixel(const Vec3f& pixel, const double alpha, const Gauss& bgd, const Gauss& fgd); //formula 14 in paper
    void setAlpha(Mat& alpha);
    void computeAlpha();
private:
    Mat trimap;
    Mat _img;
    Mat alphamap;
    vector< vector<Point> > pixelidx;
    void watchpixelidx();
};
#endif /* defined(__myGrabcut__BorderMattingHandler__) */


