#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>
#include <algorithm>

// Function declaration
std::string getImagePath();
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);

int main() {
    std::string path;
    bool flag = false;
    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));

    cv::Mat img = cv::imread(path), imgdest;
    float kernelLength, kernelWidth, normalizationFactor;
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

    std::cout << "\nYour configurations are:\nImage path: " << path
        << "\nLength = " << kernelLength
        << "\nWidth = " << kernelWidth
        << "\nNormalization factor = " << normalizationFactor;

    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;

    // Use our custom filter implementation instead of cv::filter2D
    imgdest = applyLowPassFilter(img, kernel);

    cv::imshow("Image Before", img);
    cv::imshow("Image After", imgdest);
    cv::moveWindow("Image Before", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

/**
 * Reads the entered path and formats it for opencv
 * @return The formated path
 **/
std::string getImagePath() {
    std::string path;
    std::cout << "Enter image path: ";
    std::getline(std::cin, path);
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

/**
 * Apply a low pass filter to an image using a custom convolution implementation
 * @param src The source image
 * @param kernel The convolution kernel
 * @return The filtered image
 **/
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
    /* Create output image with same size and type as input */
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    /* Identifies how many color channels the image has (1 for grayscale, 3 for RGB) */
    int channels = src.channels();


    /* Calculate padding */
    int padH = kernel.rows / 2;
    int padW = kernel.cols / 2;

    /* Create padded version of source image */
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);

    /* Apply convolution for each pixel in the image */
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            /* For each channel */
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;

                /* Apply kernel */
                for (int ky = 0; ky < kernel.rows; ky++) {
                    for (int kx = 0; kx < kernel.cols; kx++) {
                        /* Calculate position in padded image */
                        int pixelY = y + ky;
                        int pixelX = x + kx;

                        float kernelVal = kernel.at<float>(ky, kx);

                        /* Get pixel value based on number of channels */
                        float pixelVal;
                        if (channels == 1) {
                            pixelVal = static_cast<float>(padded.at<uchar>(pixelY, pixelX));
                        }
                        else {
                            pixelVal = static_cast<float>(padded.at<cv::Vec3b>(pixelY, pixelX)[c]);
                        }

                        sum += pixelVal * kernelVal;
                    }
                }

                /* Update output pixel */
                if (channels == 1) {
                    dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
                }
                else {
                    dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(sum);
                }
            }
        }
    }

    return dst;
}