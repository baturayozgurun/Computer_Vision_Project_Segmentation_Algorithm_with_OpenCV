//  Developer: Baturay Özgürün
//  Copyright © 2018 Baturay Özgürün
//  If you have any requests or questions please do not hesitate to contact me via baturay.ozgurun@gmail.com

//  Project Desciption: This code segments the object scene by implementing the Connected Component Analysis method. It also generates the gray-scale and binary images, as well as it shows the total number of segments.


//Include headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

//Include Namespaces
using namespace cv;
using namespace std;

void segmentation(){
    Mat Image, SegmentedImage,BinaryImage, SegmentedImageAppearance; //Define Mat variables
    int Ns; //Define integer variable for the total number of segments
    
    Image=imread("data/input.jpg",CV_LOAD_IMAGE_GRAYSCALE); //Read the image in grayscale
    threshold(Image, BinaryImage, 60, 255, THRESH_BINARY); //Convert the grayscale image to binary image
    //connectedComponents calculates segments of the given binary image (BinaryImage) with 8  connectedness
    Ns = connectedComponents(BinaryImage, SegmentedImage, 8, CV_32S); //It also counts total number of segments (Ns)
    normalize(SegmentedImage,SegmentedImageAppearance,0,255,NORM_MINMAX,CV_8U); //Seperate calculated segments in 8 bit scale
    applyColorMap(SegmentedImageAppearance,SegmentedImageAppearance,COLORMAP_JET); //Make colorful representation of the segmented image.
   
    imshow("Gray Scale Image", Image); //Show the original grayscale image
    imwrite("results/Gray Scale Image.jpg", Image); //Write the original grayscale image
    imshow("Binary Image", BinaryImage); //Show the binary image
    imwrite("results/Binary Image.jpg", BinaryImage); //Write the binary image
    imshow("Segmentation", SegmentedImageAppearance); //Show the colorful representation of the segmented image
    imwrite("results/Segmentation.jpg", SegmentedImageAppearance); //Write the colorful representation of the segmented image

    cout << "The total number of segments is:  " << Ns << endl; //Output of the number of segments.
    waitKey(1000); //wait to show images
}
