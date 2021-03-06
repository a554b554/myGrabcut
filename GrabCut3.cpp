#include "GrabCut.h"
GrabCut2D::~GrabCut2D(void)
{
}

static void setMask(Mat& mask,const Mat& img, Rect rect, int mode){
    if (mode == GC_WITH_RECT) {
        mask.create(img.rows, img.cols, CV_8UC1);
        mask.setTo(GC_BGD);
        mask(rect).setTo(GC_FGD);
    }
}

static void initGMM(const Mat& img,const Mat& mask, GaussianMixtureModel& fmodel, GaussianMixtureModel& bmodel){
    if (img.size()!=mask.size()) {
        cout<<"size not equal!"<<endl;
        exit(1);
    }
    fmodel.samples.clear();
    bmodel.samples.clear();
    vector<Vec3b> fsample,bsample;
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i,j)==GC_BGD||mask.at<uchar>(i,j)==GC_PR_BGD){
                bsample.push_back(img.at<Vec3b>(i,j));
            }
            else{
                fsample.push_back(img.at<Vec3b>(i,j));
            }
        }
    }
    Mat bLable,fLable; //use kmeans method to construct initial GMM.
    kmeans(bsample, GaussianMixtureModel::sizeofGMM, bLable, TermCriteria(CV_TERMCRIT_ITER, 20, 0.0), 100, KMEANS_PP_CENTERS);
    kmeans(fsample, GaussianMixtureModel::sizeofGMM, fLable, TermCriteria(CV_TERMCRIT_ITER, 20, 0.0), 100, KMEANS_PP_CENTERS);
    
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
void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    //cv::grabCut(_img, _mask, rect, _bgdModel, _fgdModel, iterCount, mode);

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
    }
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
        }
    }
}













