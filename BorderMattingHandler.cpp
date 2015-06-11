//
//  BorderMattingHandler.cpp
//  myGrabcut
//
//  Created by DarkTango on 6/2/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "BorderMattingHandler.h"

//useful function.
static double ptdist(const Point2d& p1, const Point2d& p2){
    return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

static double mul(const Point2d& p1, const Point2d& p2){
    return p1.x*p2.x+p1.y*p2.y;
}

static Scalar randomColor(){
    return Scalar(rand()*255,rand()*255,rand()*255);
}

// class function for Line.
double Line::dist(Point2d p ,bool testsign){
    Point2d vec1(dst.x-src.x, dst.y-src.y);
    Point2d vec2(p.x-src.x, p.y-src.y);
    
    
    if (mul(vec1,vec2)<=0) {
        return ptdist(p, src);
    }
    
    else if(mul( Point2d(src.x-dst.x, src.y-dst.y), Point2d(p.x-dst.x, p.y-dst.y))<=0){
        return ptdist(p, dst);
    }
    
    else{
        double ab = sqrt(vec1.x*vec1.x+vec1.y*vec1.y);
        double ac = sqrt(vec2.x*vec2.x+vec2.y*vec2.y);
        double cos = (vec1.x*vec2.x+vec1.y*vec2.y)/(ab*ac);
        return ac*sqrt(1-cos*cos);
    }
}

bool Line::isInside(Point2d p){
    Point vec1(dst.x-src.x,dst.y-src.y);
    Point vec2(p.x-src.x,p.y-src.y);
    return vec1.x*vec2.y-vec2.x*vec1.y>=0;
}


//class function for BorderMattingHandler.
BorderMattingHandler::BorderMattingHandler(const Mat& image, vector<Point>& contour){
    if (image.empty()) {
        return;
    }
    image.copyTo(_img);
    
    //init trimap
    trimap.create( _img.size(), CV_8UC1);
    Point p;
    for (p.y = 0; p.y < image.rows; p.y++) {
        for (p.x = 0; p.x < image.cols; p.x++) {
            double dist = pointPolygonTest(contour, Point2f(p.x,p.y), false);
            if (dist < 0) {
                trimap.at<uchar>(p) = BM_B;
            }
            else if (dist > 0){
                trimap.at<uchar>(p) = BM_F;
            }
            else{
                trimap.at<uchar>(p) = BM_U;
            }
        }
    }
    
    
    const int MIN_DIST = 4;
    Point cupt = contour[0];
    for (int i = 1; i < contour.size(); i++) {
        Line ln;
        ln.src = cupt;
        if (ptdist(cupt,contour[i]) > MIN_DIST) {
            ln.dst = contour[i];
            contours.push_back(ln);
            cupt = contour[i];
        }
    }
    /*for (int i = 0; i < contours.size(); i++) {
        line(image, contours[i].src, contours[i].dst, Scalar(0,0,255));
        imshow("contour", image);
        waitKey(1);
    }*/
    constructTrimap();
}

int BorderMattingHandler::minestDistanceContourIndex(Point p){
    double mindist = contours[0].dist(p);
    int index = 0;
    for (int i = 1; i < contours.size(); i++) {
        double tmpdist = contours[i].dist(p);
        if (tmpdist < mindist) {
            index = i;
            mindist = tmpdist;
        }
    }
    return index;
}

void BorderMattingHandler::constructTrimap(){
    Point p;
    double w = 6.0;
    pixelidx.clear();
    pixelidx.resize(contours.size());
    
    //traversal all pixel.
    cout<<"constructing trimap."<<endl;
    for (p.y = 0; p.y < _img.rows; p.y++) {
        for (p.x = 0; p.x < _img.cols; p.x++) {
            int id = minestDistanceContourIndex(p);
            if (contours[id].dist(p) < w) {
                pixelidx[id].push_back(p);
                trimap.at<uchar>(p) = BM_U;
            }
        }
    }
    cout<<"finished."<<endl;
    imshow("trimap", trimap);
    //watchpixelidx();
    computeAlpha();
    imshow("alpha", alphamap);
}

void BorderMattingHandler::watchpixelidx(){
    Mat img = _img.clone();
    for (int i = 0; i < pixelidx.size(); i++) {
        Scalar color = randomColor();
        for (int j = 0; j < pixelidx[i].size(); j++) {
            Point p = pixelidx[i][j];
            circle(img, p, 1, color);
        }
    }
    imshow("pixelidx", img);
}

double BorderMattingHandler::dataTermForPixel(const Vec3f &pixel, const double alpha, const Gauss &bgd, const Gauss &fgd){
    Vec3f miu;
    Mat cov(3,3,CV_64FC1);
    miu = (1-alpha)*fgd.getmean() + alpha*bgd.getmean();
    cov = (1-alpha)*(1-alpha)*fgd.getcovmat() + alpha*alpha*bgd.getcovmat();
    return Gauss::probability(miu, cov, pixel);
}

void BorderMattingHandler::computeAlpha(){
    trimap.copyTo(alphamap);
    for (int i = 0; i < pixelidx.size(); i++) {
        for (int j = 0; j < pixelidx[i].size(); j++) {
            const Point& p = pixelidx[i][j];
            double dist = contours[i].dist(p);
            if (contours[i].isInside(p)) {
                alphamap.at<uchar>(p) = 0;
            }
            else{
                alphamap.at<uchar>(p) = dist/6*255;
            }
        }
    }
}

void BorderMattingHandler::setAlpha(cv::Mat &alpha)
{
    alphamap.copyTo(alpha);
}

//class function for Gauss
Gauss::Gauss(){
    mean = Vec3f(0,0,0);
    covmat.create(3, 3, CV_64FC1);
    covmat.setTo(0);
}

double Gauss::gauss(const double center, const double width, const double r){
    if (r>center) {
        return 1;
    }
    else if (r<=0){
        return 0;
    }
    else{
        double t = (-0.5)*(r-center)*(r-center)/(width*width);
        return 1.0f/width*exp(t);
    }
}

void Gauss::addsample(Vec3f color){
    samples.push_back(color);
}

void Gauss::learn(){
    //calc mean
    Vec3f sum = 0;
    int sz = (int)samples.size();
    for (int i = 0; i < sz; i++) {
        sum += samples[i];
    }
    mean = sum/sz;
    
    //calc covmat, invcovmat and determinant.
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double sum = 0;
            for (int cnt = 0; cnt < sz; cnt++) {
                sum += (samples[cnt][i] - mean[i])*(samples[cnt][j] - mean[j]);
            }
            covmat.at<double>(i,j) = sum/sz;
        }
    }
}

double Gauss::probability(const Vec3f &mean, const cv::Mat &covmat, Vec3f color){
    double mul = 0;
    Mat miu(1,3,CV_64FC1);
    Mat ans(1,1,CV_64FC1);
    miu.at<double>(0,0) = color[0] - mean[0];
    miu.at<double>(0,1) = color[1] - mean[1];
    miu.at<double>(0,2) = color[2] - mean[2];
    ans = miu * covmat.inv() * miu.t();
    mul = (-0.5)*ans.at<double>(0,0);
    return 1.0f/sqrt(determinant(covmat))*exp(mul);
}








