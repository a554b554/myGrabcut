#include "GCApplication.h"

//Set value for the class
void GCApplication::reset()
{
	if( !mask.empty() )
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;    
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

//Set image and window name
void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
	if( _image.empty() || _winName.empty() )
		return;
	image = &_image;
	winName = &_winName;
	mask.create( image->size(), CV_8UC1);
	reset();
}

//Show the result image
void GCApplication::showImage() const
{
	if( image->empty() || winName->empty() )
		return;

	Mat res;
	Mat binMask;
	if( !isInitialized )
		image->copyTo( res );
	else
	{
		getBinMask( mask, binMask );
		image->copyTo( res, binMask );  //show the GrabCuted image
	}

	vector<Point>::const_iterator it;
	//Using four different colors show the point which have been selected
	for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )  
		circle( res, *it, radius, BLUE, thickness );
	for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )  
		circle( res, *it, radius, GREEN, thickness );
	for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
		circle( res, *it, radius, LIGHTBLUE, thickness );
	for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
		circle( res, *it, radius, PINK, thickness );

	//Draw the rectangle
	if( rectState == IN_PROCESS || rectState == SET )
		rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

	imshow( *winName, res );
}

void GCApplication::saveImage(const string name){
    if (image->empty()) {
        return;
    }
    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );  //show the GrabCuted image
        borderMatting(binMask, res);
        /*Mat rgb[3];
        split(res, rgb);
        Mat alpha(binMask.size(),CV_8UC1,Scalar(0));
        for (int i = 0; i < binMask.rows; i++) {
            for (int j = 0; j < binMask.cols; j++) {
                if (binMask.at<uchar>(i,j)==0) {
                    alpha.at<uchar>(i,j)=0;
                }
                else{
                    alpha.at<uchar>(i,j)=255;
                }
            }
        }
        Mat rgba[4]={rgb[0],rgb[1],rgb[2],alpha};
        merge(rgba, 4, res);*/
    }
        cout<<"image saved"<<endl;
    imwrite(name, res);
}

void GCApplication::borderMatting(const cv::Mat &binMask, cv::Mat &rgba){
    Mat res,tmp;
    image->copyTo(res,binMask);
    cvtColor(res, tmp, CV_BGR2GRAY);
    vector< vector <Point> > contours; // Vector for storing contour
    vector< Vec4i > hierarchy;
    int largest_contour_index=0;
    int largest_area=0;
    
    findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    //get largest contour
    for (int i = 0; i < contours.size(); i++) {
        double a = contourArea(contours[i],false);
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;
        }
    }
    
    //drawContours(res, contours, largest_contour_index, Scalar(255));
    Mat a,alpha,final;
    image->copyTo(a);
    BorderMattingHandler handler(a, contours[largest_contour_index]);
    handler.setAlpha(alpha);
    Mat rgb[3];
    split(a, rgb);
    Mat _rgba[4]={rgb[0],rgb[1],rgb[2],alpha};
    merge(_rgba, 4, rgba);
}


//Using rect initialize the pixel 
void GCApplication::setRectInMask()
{
	assert( !mask.empty() );
	mask.setTo( GC_BGD );   //GC_BGD == 0
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols-rect.x);
	rect.height = min(rect.height, image->rows-rect.y);
	(mask(rect)).setTo( Scalar(GC_PR_FGD) );    //GC_PR_FGD == 3 
}


void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if( !isPr ) //Points which are sure being FGD or BGD
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0
		fvalue = GC_FGD;    //1
	}
	else    //Probably FGD or Probably BGD
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2
		fvalue = GC_PR_FGD; //3
	}
	if( flags & BGD_KEY )
	{
		bpxls->push_back(p);
		circle( mask, p, radius, bvalue, thickness );   //Set point value = 2
	}
	if( flags & FGD_KEY )
	{
		fpxls->push_back(p);
		circle( mask, p, radius, fvalue, thickness );   //Set point value = 3
	}
}


//Mouse Click Function: flags work with CV_EVENT_FLAG 
void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN: // Set rect or GC_BGD(GC_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,
				isf = (flags & FGD_KEY) != 0;
			if( rectState == NOT_SET && !isb && !isf )//Only LEFT_KEY pressed
			{
				rectState = IN_PROCESS; //Be drawing the rectangle
				rect = Rect( x, y, 1, 1 );
			}
			if ( (isb || isf) && rectState == SET ) //Set the BGD/FGD(labels).after press the "ALT" key or "SHIFT" key,and have finish drawing the rectangle
			lblsState = IN_PROCESS;
		}
		break;
	case CV_EVENT_RBUTTONDOWN: // Set GC_PR_BGD(GC_PR_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,
				isf = (flags & FGD_KEY) != 0;
			if ( (isb || isf) && rectState == SET ) //Set the probably FGD/BGD labels
				prLblsState = IN_PROCESS;
		}
		break;
	case CV_EVENT_LBUTTONUP:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );   //After draw the rectangle
			rectState = SET;
			setRectInMask();
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();
		}
		if( lblsState == IN_PROCESS )   
		{
			setLblsInMask(flags, Point(x,y), false);    // Draw the FGD points
			lblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_RBUTTONUP:
		if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true); //Draw the BGD points
			prLblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();   //Continue showing image
		}
		else if( lblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), false);
			showImage();
		}
		else if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true);
			showImage();
		}
		break;
	}
}

//Execute GrabCut algorithm and return the iter count.
int GCApplication::nextIter()
{
	if( isInitialized )
		gc.GrabCut(*image, mask, rect, bgdModel, fgdModel,1,GC_CUT);
	else
	{
		if( rectState != SET )
			return iterCount;

		if( lblsState == SET || prLblsState == SET )
		 gc.GrabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_MASK );
		 else
		 gc.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}