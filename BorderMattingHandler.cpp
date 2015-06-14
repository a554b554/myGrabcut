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

bool Line::isOutside(Point2d p){
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
    setupGauss();
    initSolve();
    solveEnergyFunction();
    computeAlpha();
    rejectOutlier();
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

double BorderMattingHandler::dataTermForContour(const int contourid, const double sigma, const double delta){
    double ans = 0;
    for (int i = 0; i < pixelidx[contourid].size(); i++) {
        Point p = pixelidx[contourid][i];
        Vec3f color = _img.at<Vec3b>(p);
        double r = contours[contourid].dist(p);
        double alpha = Gauss::gauss(delta, sigma, r);
        ans += dataTermForPixel(color, alpha, bgdGauss[contourid], fgdGauss[contourid]);
    }
    return ans;
}

double BorderMattingHandler::smoothDifference(const double delta1, const double sigma1, const double delta2, const double sigma2)const{
    const double lamda1 = 50;
    const double lamda2 = 1000;
    double d1 = delta1 - delta2;
    double d2 = sigma1 - sigma2;
    return lamda1*d1*d1 + lamda2*d2*d2;
}

void BorderMattingHandler::computeAlpha(){
    cout<<"computing alpha..."<<endl;
    trimap.copyTo(alphamap);
    for (int i = 0; i < pixelidx.size(); i++) {
        for (int j = 0; j < pixelidx[i].size(); j++) {
            const Point& p = pixelidx[i][j];
            double dist = contours[i].dist(p);
            if (contours[i].isOutside(p)) {
                alphamap.at<uchar>(p) = 0;
            }
            else{
                //alphamap.at<uchar>(p) = Gauss::gauss(delta[i], sigma[i],contours[i].dist(p));
                alphamap.at<uchar>(p) = Gauss::gauss(5, 2,contours[i].dist(p))*255;
                cout<<Gauss::gauss(5, 2,contours[i].dist(p))*255<<endl;
            }
        }
    }
}

void BorderMattingHandler::solveEnergyFunction(){
    for (int t = 1; t < contours.size(); t++) {
        bestValue(t, delta[t-1], sigma[t-1], delta[t], sigma[t]);
    }
}

void BorderMattingHandler::initSolve(){
    cout<<"solving energy function..."<<endl;
    Gauss::discret(sigmapool, deltapool);
    sigma.clear();
    delta.clear();
    sigma.resize(contours.size());
    delta.resize(contours.size());
    int bestsigmaid = 0;
    int bestdeltaid = 0;
    double mincost = INFINITY;
    for (int i = 0; i < deltapool.size(); i++) {
        for (int j = 0; j < sigmapool.size(); j++) {
            if (sigmapool[j] > deltapool[i]) {
                break;
            }
            double tmpcost = dataTermForContour(0, sigmapool[j], deltapool[i]);
            if (tmpcost < mincost) {
                mincost = tmpcost;
                bestsigmaid = j;
                bestdeltaid = i;
            }
        }
    }
    sigma[0] = sigmapool[bestsigmaid];
    delta[0] = deltapool[bestdeltaid];
}

void BorderMattingHandler::setupGauss(){
    const int L = 41;
    bgdGauss.clear();
    fgdGauss.clear();
    cout<<"setting up gaussian model..."<<endl;
    for (int i = 0; i < contours.size(); i++) {
        Point p((contours[i].src.x+contours[i].dst.x)/2, (contours[i].src.y+contours[i].dst.y)/2);
        int left = (p.x-L>=0)?p.x-L:0;
        int right = (p.x+L<_img.cols)?p.x+L:_img.cols-1;
        int low = (p.y-L>=0)?p.y-L:0;
        int up = (p.y+L<_img.rows)?p.y+L:_img.rows-1;
        Gauss _bgd,_fgd;
        for (p.y = low; p.y <= up; p.y++) {
            for (p.x = left; p.x <= right; p.x++){
                if (trimap.at<uchar>(p) == BM_B) {
                    _bgd.addsample(_img.at<Vec3b>(p));
                }
                else if (trimap.at<uchar>(p) == BM_F){
                    _fgd.addsample(_img.at<Vec3b>(p));
                }
            }
        }
        _bgd.learn();
        _fgd.learn();
        bgdGauss.push_back(_bgd);
        fgdGauss.push_back(_fgd);
    }
}

void BorderMattingHandler::bestValue(const int contourid, const double delta0, const double sigma0, double &delta, double &sigma){
    int bestdeltaid = 0;
    int bestsigmaid = 0;
    double mincost = INFINITY;
    for (int i = 0; i < deltapool.size(); i++) {
        for (int j = 0; j < sigmapool.size(); j++) {
            if (sigmapool[j] > deltapool[i]) {
                break;
            }
            double tmpcost = dataTermForContour(contourid, sigmapool[j], deltapool[i])+smoothDifference(delta0, sigma0, deltapool[i], sigmapool[j]);
            if (tmpcost < mincost) {
                mincost = tmpcost;
                bestdeltaid = i;
                bestsigmaid = j;
            }
        }
    }
    delta = deltapool[bestdeltaid];
    sigma = sigmapool[bestsigmaid];
}


void BorderMattingHandler::setAlpha(cv::Mat &alpha)
{
    alphamap.copyTo(alpha);
}

void BorderMattingHandler::rejectOutlier(){
    vector< vector <Point> > contours; // Vector for storing contour
    vector< Vec4i > hierarchy;
    int largest_contour_index=0;
    int largest_area=0;
    Mat _alpha = alphamap.clone();
    findContours(_alpha, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    //get largest contour
    for (int i = 0; i < contours.size(); i++) {
        double a = contourArea(contours[i],false);
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;
        }
    }
    Point p;
    for (p.y = 0; p.y<alphamap.rows; p.y++) {
        for (p.x = 0; p.x<alphamap.cols; p.x++) {
            if (pointPolygonTest(contours[largest_contour_index], p, false)<0){
                alphamap.at<uchar>(p) = 0;
            }
        }
    }
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
   /* else if (r<=0){
        return 0;
    }*/
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

void Gauss::discret(vector<double> &sigma, vector<double> &delta){
    sigma.clear();
    delta.clear();
    for (double i = 0.1; i <= 6; i += (6.0/30)) {
        delta.push_back(i);
    }
    for (double i = 0.1; i <= 2; i += (2.0/10)) {
        sigma.push_back(i);
    }
}






