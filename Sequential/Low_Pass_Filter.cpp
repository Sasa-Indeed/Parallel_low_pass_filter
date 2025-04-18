#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>


int main() {

    cv::Mat img = cv::imread("D:/ASU/5-Senior 2/Spring 25/CSE455 HPC/Project/lena.png"), imgdest;

    float kernelLength, kernelWidth , normalizationFactor;

    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;
    std::cout << "Choose the normalization factor for the default value enter 0: ";
    std::cin >> normalizationFactor;
    
    if (normalizationFactor <= 0) {
        normalizationFactor = kernelLength * kernelWidth;
    }

    std::cout << "\nYour configurations are\nLength = " << kernelLength << "\nWidth = " <<  kernelWidth << "\nNormalization factor = " << normalizationFactor;
    
    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;
    
    cv::filter2D(img, imgdest, CV_8U, kernel);

    
    cv::imshow("Image Before", img);
    cv::imshow("Image After", imgdest);
    cv::moveWindow("Image Before", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}