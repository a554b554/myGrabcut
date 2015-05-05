//
//  Grabcut.cpp
//  myGrabcut
//
//  Created by DarkTango on 4/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "GrabCut.h"

GrabCut2D::~GrabCut2D(void)
{
}

GaussianMixtureModel::GaussianMixtureModel(Mat& _params){//13*1
    const int modelsize = 13;
    if (_params.empty()) {
        _params.create(1, modelsize*sizeofGMM, CV_64FC1);
        _params.setTo(Scalar(0));
    }
    //cout<<"params: "<<_params<<endl;
    for (int i = 0; i < sizeofGMM; i++) {
        weight[i] = _params.at<double>(0,i);
        mean[i][0] = _params.at<double>(0,3*i+sizeofGMM);
        mean[i][1] = _params.at<double>(0,3*i+sizeofGMM+1);
        mean[i][2] = _params.at<double>(0,3*i+sizeofGMM+2);
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                cov[i][x][y] = _params.at<double>(0,9*i+sizeofGMM+3*sizeofGMM+x*3+y);
            }
        }
    }
    calcDetandInv();
}

static void setMask(Mat& mask,const Mat& img, Rect rect, int mode){
    if (mode == GC_WITH_RECT) {
        mask.create(img.rows, img.cols, CV_8UC1);
        mask.setTo(GC_BGD);
        mask(rect).setTo(GC_PR_FGD);
        cout<<"CG_WITH_RECT"<<endl;
    }
    else{ //GC_WITH_MASK
        cout<<"GC_WITH_MASK"<<endl;
        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                Point pt(j,i);
                if (!rect.contains(pt)) {
                    mask.at<uchar>(i,j) = GC_BGD;
                }
                else{
                    if (mask.at<uchar>(i,j) != GC_FGD && mask.at<uchar>(i,j)!= GC_BGD) {
                        mask.at<uchar>(i,j) = GC_PR_FGD;
                    }
                }
            }
        }
    }
}


static double colordiffnorm(Vec3b color1, Vec3b color2){
    Vec3d colordiff = static_cast<Vec3d>(color1) -static_cast<Vec3d>(color2);
    double diff[3];
    diff[0] = colordiff[0];
    diff[1] = colordiff[1];
    diff[2] = colordiff[2];
    return sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
}


static void initGMM(const Mat& img,const Mat& mask, GaussianMixtureModel& fmodel, GaussianMixtureModel& bmodel){
    if (img.size()!=mask.size()) {
        cout<<"size not equal!"<<endl;
        exit(1);
    }
    fmodel.samples.clear();
    bmodel.samples.clear();
    vector<Vec3f> fsample,bsample;
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i,j)==GC_BGD||mask.at<uchar>(i,j)==GC_PR_BGD){
                bsample.push_back((Vec3f)img.at<Vec3b>(i,j));
            }
            else{
                fsample.push_back((Vec3f)img.at<Vec3b>(i,j));
            }
        }
    }
    Mat bLable,fLable; //use kmeans method to construct initial GMM.
    Mat _bsample((int)bsample.size(),3,CV_32FC1,&bsample[0][0]);
    Mat _fsample((int)fsample.size(),3,CV_32FC1,&fsample[0][0]);
    kmeans(_bsample, GaussianMixtureModel::sizeofGMM, bLable, TermCriteria(CV_TERMCRIT_ITER, 20, 0.0), 0, KMEANS_PP_CENTERS);
    kmeans(_fsample, GaussianMixtureModel::sizeofGMM, fLable, TermCriteria(CV_TERMCRIT_ITER, 20, 0.0), 0, KMEANS_PP_CENTERS);
    
    fmodel.samples.resize(GaussianMixtureModel::sizeofGMM);
    bmodel.samples.resize(GaussianMixtureModel::sizeofGMM);
    //init
    for (int i = 0; i < GaussianMixtureModel::sizeofGMM; i++) {
        fmodel.samples[i].clear();
        bmodel.samples[i].clear();
    }
    for (int i = 0; i < bLable.rows; i++) {
        int Cid = bLable.at<int>(i,0);
        bmodel.samples[Cid].push_back(bsample[i]);
    }
    for (int i = 0; i < fLable.rows; i++) {
        int Cid = fLable.at<int>(i,0);
        fmodel.samples[Cid].push_back(fsample[i]);
    }
}

double computeBeta(const Mat& img){
    double ans = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d color = img.at<Vec3b>(i,j);
            if ( j > 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(i,j-1);
                ans += diff.dot(diff);
            }
            if (i > 0 && j >0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(i-1,j-1);
                ans += diff.dot(diff);
            }
            if (i > 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(i-1,j);
                ans += diff.dot(diff);
            }
            if (i > 0 && j < img.cols - 1) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(i-1,j+1);
                ans += diff.dot(diff);
            }
        }
    }
    if (ans <= numeric_limits<double>::epsilon()) {
        ans = 0;
    }
    else{
        ans = 1.0/(2*ans/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2));
    }
    return ans;
}

static void assignGMMcomponents(const Mat& img, const Mat& mask, GaussianMixtureModel& fGMM, GaussianMixtureModel& bGMM){
    fGMM.samples.clear();
    bGMM.samples.clear();
    const int gmmsize = GaussianMixtureModel::sizeofGMM;
    fGMM.samples.resize(gmmsize);
    bGMM.samples.resize(gmmsize);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i,j);
            if (mask.at<uchar>(i,j) == GC_BGD) {
                bGMM.addSample(color);
                continue;
            }
            if (mask.at<uchar>(i,j) == GC_FGD) {
                fGMM.addSample(color);
                continue;
            }
            if (bGMM.probability(color)>fGMM.probability(color)) {
                bGMM.addSample(color);
            }
            else{
                fGMM.addSample(color);
            }
        }
    }
    fGMM.learning();
    bGMM.learning();
}

// use the gcgraph library to construct graph
static void constructGraph(const Mat& img, const Mat& mask, GaussianMixtureModel& bGMM, GaussianMixtureModel& fGMM, const Mat& weightmatH, const Mat& weightmatV,const Mat& weightmatUL, const Mat& weightmatUR, GCGraph<double>& graph, const double lambda){
    int vtxSize = img.cols*img.rows;
    int edgeSize = 2*(4*vtxSize - 3*(img.cols + img.rows) + 2);
    graph.create(vtxSize, edgeSize);
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            //init
            int vid = graph.addVtx();
            double source,sink;
            Vec3b color = img.at<Vec3b>(i,j);
            
            
            //set t-weight
            if (mask.at<uchar>(i,j)==GC_PR_FGD||mask.at<uchar>(i,j)==GC_PR_BGD) {
                source = -log(bGMM.probability(color)); //if source > sink, the cut will cut sink, which means pixel is foreground
                sink = -log(fGMM.probability(color));
            }
            else if (mask.at<uchar>(i,j)==GC_BGD){
                source = 0;
                sink = lambda;
            }
            else if (mask.at<uchar>(i,j)==GC_FGD){
                source = lambda;
                sink = 0;
            }
            graph.addTermWeights(vid, source, sink);
            
            //set n-weight
            if (j > 0) { // Horizontal
                double weight = weightmatH.at<double>(i,j-1);
                graph.addEdges(vid, vid-1, weight, weight);
            }
            if (i > 0) { // Vertical
                double weight = weightmatV.at<double>(i-1,j);
                graph.addEdges(vid, vid-img.cols, weight, weight);
            }
            if (i > 0 && j < img.cols-1) { //upright
                double weight = weightmatUR.at<double>(i-1,j);
                graph.addEdges(vid, vid-img.cols+1, weight, weight);
            }
            if (j > 0 && i > 0) { //upleft
                double weight = weightmatUL.at<double>(i-1,j-1);
                graph.addEdges(vid, vid-img.cols-1, weight, weight);
            }
        }
    }
    
}

int GaussianMixtureModel::maxComponent(const Vec3b BGR){
    int compid = 0;
    double maxprob = 0;
    for (int i = 0; i < sizeofGMM; i++) {
        double prob = probabilityInComponent(i, BGR);
        if ( prob > maxprob) {
            compid = i;
            maxprob = prob;
        }
    }
    return compid;
}

static void segmentation(GCGraph<double>& graph, Mat& mask){
    graph.maxFlow();
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i,j) == GC_PR_BGD || mask.at<uchar>(i,j) == GC_PR_FGD) {
                if (graph.inSourceSegment(i*mask.cols+j)) {
                    mask.at<uchar>(i,j) = GC_PR_FGD;
                }
                else{
                    mask.at<uchar>(i,j) = GC_PR_BGD;
                }
            }
        }
    }
}

static void calcEdgeWeight(const Mat& img, Mat& weightmatH, Mat& weightmatV, Mat& weightmatUL, Mat& weightmatUR, const double gamma, const double beta){
    weightmatH.create(img.rows, img.cols-1, CV_64FC1);
    weightmatV.create(img.rows-1, img.cols, CV_64FC1);
    weightmatUL.create(img.rows-1, img.cols-1, CV_64FC1);
    weightmatUR.create(img.rows-1, img.cols-1, CV_64FC1);
    weightmatH.setTo(Scalar(0));
    weightmatV.setTo(Scalar(0));
    weightmatUL.setTo(Scalar(0));
    weightmatUR.setTo(Scalar(0));
    const double gammadivsqrt2 = gamma/sqrt(2.0);
    
    //calculate Horizontal weight map.
    for (int i = 0; i < weightmatH.rows; i++) {
        for (int j = 0; j < weightmatH.cols; j++) {
            weightmatH.at<double>(i,j) = gamma*exp(-beta*colordiffnorm(img.at<Vec3b>(i,j+1),img.at<Vec3b>(i,j)));
        }
    }
    
    //calculate Vertical weight map.
    for (int i = 0; i < weightmatV.rows; i++) {
        for (int j = 0; j < weightmatV.cols; j++) {
            weightmatV.at<double>(i,j) = gamma*exp(-beta*colordiffnorm(img.at<Vec3b>(i+1,j),img.at<Vec3b>(i,j)));
        }
    }
    
    //calculate upleft weight map.
    for (int i = 0; i < weightmatUL.rows; i++) {
        for (int j = 0; j < weightmatUL.cols; j++) {
            weightmatUL.at<double>(i,j) = gammadivsqrt2*exp(-beta*colordiffnorm(img.at<Vec3b>(i+1,j+1),img.at<Vec3b>(i,j)));
        }
    }
    
    //calculate upright weight map.
    for (int i = 0; i < weightmatUR.rows; i++) {
        for (int j = 0; j < weightmatUR.cols; j++) {
            weightmatUR.at<double>(i,j) = gammadivsqrt2*exp(-beta*colordiffnorm(img.at<Vec3b>(i+1,j), img.at<Vec3b>(i,j+1)));
        }
    }
}

void GaussianMixtureModel::addSample(Vec3b BGR){
    int compid = maxComponent(BGR);
    samples[compid].push_back(BGR);
}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    cv::grabCut(_img, _mask, rect, _bgdModel, _fgdModel, iterCount, mode);
    return;
    
    cout<<"mask: "<<_mask.getMatRef()<<endl;
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    GaussianMixtureModel bGMM(bgdModel);
    GaussianMixtureModel fGMM(fgdModel);
    
    if (img.empty()) {
        cout<<"empty image!!!"<<endl;
        exit(1);
    }
    if (mode != GC_CUT) {
        setMask(mask,img,rect,mode);
        initGMM(img,mask,fGMM,bGMM);
        fGMM.learning();
        bGMM.learning();
    }
    
    
    const double gamma = 50;
    const double lambda = 450;
    const double beta = computeBeta(img);
    Mat weightmatH,weightmatV,weightmatUR,weightmatUL;
    int64 t0 = getTickCount();
    calcEdgeWeight(img, weightmatH, weightmatV, weightmatUL, weightmatUR, gamma, beta);
    cout<<"time for calcedgeweight: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    t0 = getTickCount();
    for (int i = 0; i < iterCount; i++) {
        assignGMMcomponents(img, mask, fGMM, bGMM);
        cout<<"time for assignGMM components: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
        t0 = getTickCount();
        GCGraph<double> graph;
        constructGraph(img, mask, bGMM, fGMM, weightmatH, weightmatV, weightmatUL, weightmatUR, graph, lambda);
        cout<<"time for constructGraph: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
        t0 = getTickCount();
        segmentation(graph, mask);
        cout<<"time for segmentation: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    }
    //cout<<"output mask: "<<mask<<endl;
}


int GaussianMixtureModel::totalsize() const{
    int ans = 0;
    for (int i = 0; i < GaussianMixtureModel::sizeofGMM; i++) {
        ans += samples[i].size();
    }
    return ans;
}

void GaussianMixtureModel::learning(){
    int total = totalsize();
    int gsize = GaussianMixtureModel::sizeofGMM;
    // init all parameters
    //learning weight
    for (int i = 0; i < gsize; i++) {
        weight[i] = samples[i].size()*1.0/total;
    }
    //learning mean
    for (int i = 0; i < gsize; i++) {
        double sumb = 0;
        double sumg = 0;
        double sumr = 0;
        for (int j = 0; j < samples[i].size(); j++) {
            sumb += static_cast<double>(samples[i][j][0]);
            sumg += static_cast<double>(samples[i][j][1]);
            sumr += static_cast<double>(samples[i][j][2]);
        }
        mean[i][0] = sumb/samples[i].size();
        mean[i][1] = sumg/samples[i].size();
        mean[i][2] = sumr/samples[i].size();
    }
    
    //learning cov
    for (int i = 0; i < gsize; i++) {
        double sum[3][3];
        memset(sum, 0, sizeof(double)*3*3);
        for (int j = 0; j < samples[i].size(); j++) {
            sum[0][0] += (samples[i][j][0]-mean[i][0])*(samples[i][j][0]-mean[i][0]);
            sum[0][1] += (samples[i][j][0]-mean[i][0])*(samples[i][j][1]-mean[i][1]);
            sum[0][2] += (samples[i][j][0]-mean[i][0])*(samples[i][j][2]-mean[i][2]);
            sum[1][1] += (samples[i][j][1]-mean[i][1])*(samples[i][j][1]-mean[i][1]);
            sum[1][2] += (samples[i][j][1]-mean[i][1])*(samples[i][j][2]-mean[i][2]);
            sum[2][2] += (samples[i][j][2]-mean[i][2])*(samples[i][j][2]-mean[i][2]);
        }
        double n = samples[i].size()-1;
        cov[i][0][0] = sum[0][0]/n;
        cov[i][0][1] = cov[i][1][0] = sum[0][1]/n;
        cov[i][0][2] = cov[i][2][0] = sum[0][2]/n;
        cov[i][1][1] = sum[1][1]/n;
        cov[i][1][2] = cov[i][2][1] = sum[1][2]/n;
        cov[i][2][2] = sum[2][2]/n;
    }
    
    //calc determinant and inv
    calcDetandInv();
}

void GaussianMixtureModel::calcDetandInv(){
    for (int i = 0; i < sizeofGMM; i++) {
        Mat covMat(3,3,CV_64FC1);
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                covMat.at<double>(j,k) = cov[i][j][k];
            }
        }
        covInv[i] = covMat.inv();
        covDeterminant[i] = determinant(covMat);
    }
}

double GaussianMixtureModel::probabilityInComponent(int component, const Vec3b BGR){
    Mat diff(3,1,CV_64FC1);
    diff.at<double>(0,0) = BGR[0]-mean[component][0];
    diff.at<double>(1,0) = BGR[1]-mean[component][1];
    diff.at<double>(2,0) = BGR[2]-mean[component][2];
    Mat tmp = diff.t()*covInv[component]*diff;
    double mul = -0.5*tmp.at<double>(0,0);
    return 1.0/sqrt(covDeterminant[component])*exp(mul);
}

double GaussianMixtureModel::probability(const Vec3b BGR){
    double ans = 0;
    for (int i = 0; i < sizeofGMM; i++) {
        ans += weight[i]*probabilityInComponent(i, BGR);
    }
    return ans;
}







