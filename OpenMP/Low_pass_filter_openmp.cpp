#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>

std::string getImagePath();
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);

int main() {
    std::string path;
    bool flag = false;
    int threadID;
    std::string grayFlag = "y";

    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    std::cout << "Do want the image in gray scale?[y/n]: y\n";
    std::cin >> grayFlag;

    if (grayFlag[0] == 'y') {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    float kernelLength, kernelWidth, normalizationFactor;

    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;
    std::cout << "Choose the normalization factor (or 0 for default): ";
    std::cin >> normalizationFactor;

    if (normalizationFactor <= 0) {
        normalizationFactor = kernelLength * kernelWidth;
    }

    std::cout << "\nYour configurations are:\nImage path: " << path
        << "\nLength = " << kernelLength
        << "\nWidth = " << kernelWidth
        << "\nNormalization factor = " << normalizationFactor << "\n";

    double start_time = omp_get_wtime();

    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;

    cv::Mat imgdest = cv::Mat::zeros(img.size(), img.type());

    int totalRows = img.rows;

#pragma omp parallel private(threadID)
    {
        int numThreads = omp_get_num_threads();
        threadID = omp_get_thread_num();

        int rowsPerThread = totalRows / numThreads;
        int startRow = threadID * rowsPerThread;
        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;


        int pad = static_cast<int>(kernelLength / 2);
        int safeStart = std::max(startRow - pad, 0);
        int safeEnd = std::min(endRow + pad, totalRows);

        cv::Mat subImage = img.rowRange(safeStart, safeEnd);
        cv::Mat subDest;


        subDest = applyLowPassFilter(subImage, kernel);

        int copyStart = startRow - safeStart;
        int copyLength = endRow - startRow;
        subDest.rowRange(copyStart, copyStart + copyLength).copyTo(imgdest.rowRange(startRow, endRow));
    }

    double end_time = omp_get_wtime();
    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;

    cv::imshow("Image Before", img);
    cv::imshow("Image After (Parallel Filter)", imgdest);
    cv::moveWindow("Image Before", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

std::string getImagePath() {
    std::string path;
    std::cout << "Enter image path: ";
    std::getline(std::cin, path);
    std::replace(path.begin(), path.end(), '\\', '/');
    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
    return path;
}

/**
 * Apply a low pass filter to an image using a custom convolution implementation
 * @param src The source image
 * @param kernel The convolution kernel
 * @return The filtered image
 **/
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
    double start_time = omp_get_wtime();

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

    /* Cache kernel values to avoid repeated access */
    std::vector<float> kernelValues(kernel.rows * kernel.cols);
    for (int ky = 0; ky < kernel.rows; ky++) {
        for (int kx = 0; kx < kernel.cols; kx++) {
            kernelValues[ky * kernel.cols + kx] = kernel.at<float>(ky, kx);
        }
    }

    /* Apply convolution for each pixel in the image using OpenMP */
    if (channels == 1) {
        /* Grayscale processing */
#pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                float sum = 0.0;
                /* Apply kernel */
                for (int ky = 0; ky < kernel.rows; ky++) {
                    for (int kx = 0; kx < kernel.cols; kx++) {
                        /* Calculate position in padded image */
                        int pixelY = y + ky;
                        int pixelX = x + kx;
                        sum += static_cast<float>(padded.at<uchar>(pixelY, pixelX)) *
                            kernelValues[ky * kernel.cols + kx];
                    }
                }
                /* Update output pixel */
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
            }
        }
    }
    else {
        /* Color image processing */
#pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                float sum[3] = { 0.0, 0.0, 0.0 };
                /* Apply kernel */
                for (int ky = 0; ky < kernel.rows; ky++) {
                    for (int kx = 0; kx < kernel.cols; kx++) {
                        /* Calculate position in padded image */
                        int pixelY = y + ky;
                        int pixelX = x + kx;
                        float kernelVal = kernelValues[ky * kernel.cols + kx];
                        cv::Vec3b pixel = padded.at<cv::Vec3b>(pixelY, pixelX);

                        /* Process all channels together for better cache efficiency */
                        sum[0] += static_cast<float>(pixel[0]) * kernelVal;
                        sum[1] += static_cast<float>(pixel[1]) * kernelVal;
                        sum[2] += static_cast<float>(pixel[2]) * kernelVal;
                    }
                }
                /* Update output pixel */
                cv::Vec3b& outPixel = dst.at<cv::Vec3b>(y, x);
                outPixel[0] = cv::saturate_cast<uchar>(sum[0]);
                outPixel[1] = cv::saturate_cast<uchar>(sum[1]);
                outPixel[2] = cv::saturate_cast<uchar>(sum[2]);
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Filter function execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;

    return dst;
}