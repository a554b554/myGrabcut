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


//this function init each GMM by kmeans.
static void initGMM(GMM& fgdGMM, GMM& bgdGMM,const Mat& img,const Mat& mask){
    Mat fgdlabel,bgdlabel;
   /* vector<Vec3f> fgdsample,bgdsample;
    for (int i  = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            uchar msk = mask.at<uchar>(i,j);
            if (msk == GC_BGD || msk == GC_PR_BGD) {
                bgdsample.push_back(img.at<Vec3b>(i,j));
            }
            else{
                fgdsample.push_back(img.at<Vec3b>(i,j));
            }
        }
    }
    Mat _bgdsample((int)bgdsample.size(), 3, CV_32FC1, &bgdsample[0][0]);
    kmeans(_bgdsample, GMM::componentSize, bgdlabel, TermCriteria( CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
    Mat _fgdsample((int)fgdsample.size(), 3, CV_32FC1, &fgdsample[0][0]);
    kmeans(_fgdsample, GMM::componentSize, fgdlabel, TermCriteria( CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
    */
    
    const int kMeansItCount = 10;  //iter count
    const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii
    
    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentSize, bgdLabels,
           TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentSize, fgdLabels,
           TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    fgdGMM.init();
    for (int i = 0; i < fgdSamples.size(); i++) {
        fgdGMM.addsample(fgdSamples[i], fgdLabels.at<int>(i,0));
    }
    fgdGMM.learn();
    fgdGMM.printinfo(PRINT_ALL);
    bgdGMM.init();
    for (int i = 0; i < bgdSamples.size(); i++) {
        bgdGMM.addsample(bgdSamples[i], bgdLabels.at<int>(i,0));
    }
    bgdGMM.learn();
    bgdGMM.printinfo(PRINT_ALL);
}

//this function assign sample to each component.
static void assignGMMComponent(const Mat& img, const Mat& mask, GMM& fgdGMM, GMM& bgdGMM){
    fgdGMM.init();
    bgdGMM.init();
    Point p;
    // traversal image
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3f color = static_cast<Vec3f>(img.at<Vec3b>(p));
            uchar msk = mask.at<uchar>(p);
            if (msk == GC_BGD|| msk == GC_PR_BGD) {
                bgdGMM.addsample(color);
            }
            else if(msk == GC_FGD || msk == GC_PR_FGD){
                fgdGMM.addsample(color);
            }
        }
    }
    fgdGMM.learn();
    bgdGMM.learn();
    
}

static double calcbeta(const Mat& img){
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 )
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2));
    return beta;
}

static void calcnlink(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta){
    const double gamma = 50;
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else{
                leftW.at<double>(y,x) = 0;
            }
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else{
                upleftW.at<double>(y,x) = 0;
            }
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else{
                upW.at<double>(y,x) = 0;
            }
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else{
                uprightW.at<double>(y,x) = 0;
            }
        }
    }
}

static void graphCut(const Mat& img, Mat& mask, GMM& fgdGMM, GMM& bgdGMM, Mat&leftW, Mat& upleftW, Mat& upW, Mat& uprightW){
    typedef Graph<double, double, double> GraphType;
    const double gamma = 50;
    const double lambda = 9*gamma;
    int nodesize = img.rows * img.cols;
    int edgesize = img.rows*(img.cols-1) + (img.rows-1)*img.cols + (img.rows-1)*(img.cols-1)*2;
    GraphType* graph = new GraphType(nodesize,edgesize);
    
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3f color = static_cast<Vec3f>(img.at<Vec3b>(p));
            double source,terminal;  //source is foreground, terminal is background
            uchar msk = mask.at<uchar>(p);
            int vtxID = graph->add_node(); // return the index of vertex.
            
            //set t-link
            if (msk == GC_PR_FGD || msk == GC_PR_BGD) {
                source = -log(bgdGMM.posibility(color));
                terminal = -log(fgdGMM.posibility(color));
            }
            else if (msk == GC_BGD){
                source = 0;
                terminal = lambda;
            }
            else{ //GC_FGD
                source = lambda;
                terminal = 0;
            }
            graph->add_tweights(vtxID, source, terminal);
            
            //set n-link
            if (p.x>0) {
                double w = leftW.at<double>(p);
                graph->add_edge(vtxID, vtxID-1, w, w);
            }
            if (p.x>0 && p.y>0) {
                double w = upleftW.at<double>(p);
                graph->add_edge(vtxID, vtxID-img.cols-1, w, w);
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph->add_edge( vtxID, vtxID-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph->add_edge( vtxID, vtxID-img.cols+1, w, w );
            }
        }
    }
    
    //use graph cut
    double flow = graph->maxflow();
    cout<<"max flow: "<<flow<<endl;

    //set mask
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            int vtxID = p.x + p.y * mask.cols;
            if (mask.at<uchar>(p)==GC_PR_BGD||mask.at<uchar>(p)==GC_PR_FGD) {
                if (graph->what_segment(vtxID) == GraphType::SOURCE) {
                    mask.at<uchar>(p) = GC_PR_FGD;
                }
                else{
                    mask.at<uchar>(p) = GC_PR_BGD;
                }
            }
        }
    }
}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    cv::grabCut(_img, _mask, rect, _bgdModel, _fgdModel, iterCount, mode);
    return;
    GMM fgdGMM(_fgdModel.getMatRef());
    GMM bgdGMM(_bgdModel.getMatRef());
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    initGMM(fgdGMM, bgdGMM, img, mask);
    Mat leftW, upleftW, upW, uprightW;
    calcnlink(img, leftW, upleftW, upW, uprightW, calcbeta(img));
    for (int i = 0; i < iterCount; i++) {
        cout<<"aasigning GMM component..."<<endl;
        assignGMMComponent(img, mask, fgdGMM, bgdGMM);
        cout<<"begin graph cut..."<<endl;
        graphCut(img, mask, fgdGMM, bgdGMM, leftW, upleftW, upW, uprightW);
    }
}

GMM::GMM(Mat& _model){
    const int mdsize = 13;
    if (_model.empty()) {
        _model.create(1, mdsize*componentSize, CV_64FC1);
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != mdsize * componentSize)){
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );
    }
    weight = _model.ptr<double>(0);
    mean = weight + componentSize;
    cov = mean + 3*componentSize;
    
    covmat.resize(componentSize);
    invcovmat.resize(componentSize);
    samples.resize(componentSize);
    for (int i = 0; i < componentSize; i++) {
        covmat[i].create(3, 3, CV_64FC1);
        invcovmat[i].create(3, 3, CV_64FC1);
    }
    
    for (int i = 0; i < componentSize; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                covmat[i].at<double>(j,k) = cov[i*9+j*3+k];
            }
        }
        invcovmat[i] = covmat[i].inv();
        covdet[i] = determinant(covmat[i]);
        meancolor[i] = Vec3f(mean[i*3],mean[i*3+1],mean[i*3+2]);
    }
    //cout<<covmat[0];
}

void GMM::init(){
    samples.clear();
    samples.resize(componentSize);
}

void GMM::addsample(Vec3f color, int which){
    samples[which].push_back(color);
}

void GMM::learn(){
    //calc weight
    int total = 0;
    for (int i = 0; i < componentSize; i++) {
        total += samples[i].size();
    }
    for (int i = 0; i < componentSize; i++) {
        weight[i] = (double)samples[i].size()/total;
    }
    
    
    //calc mean
    for (int i = 0; i < componentSize; i++) {
        Vec3f sum = 0;
        for (int j = 0; j < samples[i].size(); j++) {
            sum += samples[i][j];
        }
        sum = sum/(int)samples[i].size();
        mean[i*3] = sum[0];
        mean[i*3+1] = sum[1];
        mean[i*3+2] = sum[2];
        meancolor[i]=Vec3f(sum[0],sum[1],sum[2]);
    }
    
    //calc covmat, invcovmat and determinant.
    for (int i = 0; i < componentSize; i++) {
        /*Vec3f sum = 0;
        for (int j = 0; j < samples[i].size(); j++) {
            sum += samples[i][j] - meancolor[i];
        }*/
    
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                double sum = 0;
                for (int cnt = 0; cnt < samples[i].size(); cnt++) {
                    sum += (samples[i][cnt][j] - meancolor[i][j])*(samples[i][cnt][k] - meancolor[i][k]);
                }
                covmat[i].at<double>(j,k) = sum/samples[i].size();
            }
        }
        invcovmat[i] = covmat[i].inv();
        covdet[i] = determinant(covmat[i]);
    }
}

double GMM::posibility(Vec3f color){
    double ans = 0;
    for (int i = 0; i < componentSize; i++) {
        ans += weight[i]*posibility(color, i);
    }
    return ans;
}

double GMM::posibility(Vec3f color, int cid){
    double mul = 0;
    Mat miu(1,3,CV_64FC1);
    Mat ans(1,1,CV_64FC1);
    miu.at<double>(0,0) = color[0]-meancolor[cid][0];
    miu.at<double>(0,1) = color[1]-meancolor[cid][1];
    miu.at<double>(0,2) = color[2]-meancolor[cid][2];
    ans = miu * invcovmat[cid] * miu.t();
    mul = (-0.5)*ans.at<double>(0,0);
    return 1.0f/sqrt(covdet[cid])*exp(mul);
}

int GMM::whichComponent(Vec3f color){
    int com = 0;
    double maxpossibility = -1;
    for (int i = 0; i < componentSize; i++) {
        double poss = posibility(color, i);
        if (poss > maxpossibility) {
            maxpossibility = poss;
            com = i;
        }
    }
    return com;
}

void GMM::addsample(Vec3f color){
    addsample(color, whichComponent(color));
}

void GMM::printinfo(int printtype){
    if (printtype == PRINT_ALL || printtype == PRINT_WEIGHT) {
        for (int i = 0; i < componentSize; i++) {
            cout<<"weight "<<i<<" "<<weight[i]<<endl;
        }
        cout<<endl;
    }
    if (printtype == PRINT_ALL || printtype == PRINT_MEAN) {
        for (int i = 0; i < componentSize; i++) {
            cout<<"mean "<<i<<" "<<meancolor[i]<<endl;
        }
        cout<<endl;
    }
    if (printtype == PRINT_ALL || printtype == PRINT_COV) {
        for (int i = 0; i < componentSize; i++) {
            cout<<"cov "<<i<<" "<<covmat[i]<<endl;
        }
        cout<<endl;
    }
    if (printtype == PRINT_ALL || printtype == PRINT_INVCOV) {
        for (int i = 0; i < componentSize; i++) {
            cout<<"invcov "<<i<<" "<<invcovmat[i]<<endl;
        }
        cout<<endl;
    }
    if (printtype == PRINT_ALL || printtype == PRINT_DET) {
        for (int i = 0; i < componentSize; i++) {
            cout<<"covdet "<<i<<" "<<covdet[i]<<endl;
        }
        cout<<endl;
    }
}







