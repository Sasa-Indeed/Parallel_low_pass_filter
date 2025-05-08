//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <omp.h>
//
//std::string getImagePath();
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//
//    // Get valid image path
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    cv::Mat img = cv::imread(path);
//    if (img.empty()) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    // Option to convert to grayscale
//    std::cout << "Do you want the image in grayscale? [y/n]: ";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
//    }
//
//    int kernelLength, kernelWidth;
//
//    // Get kernel dimensions
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//
//    std::cout << "\nYour configurations are:\n"
//              << "Image path: " << path << "\n"
//              << "Length = " << kernelLength << "\n"
//              << "Width = " << kernelWidth << "\n";
//
//    double start_time = omp_get_wtime();
//
//    // Create separable 1D kernels, normalized to sum to 1
//    cv::Mat horizontalKernel = cv::Mat::ones(1, kernelWidth, CV_32F) / kernelWidth;
//    cv::Mat verticalKernel = cv::Mat::ones(kernelLength, 1, CV_32F) / kernelLength;
//
//    cv::Mat tempImg = cv::Mat::zeros(img.size(), img.type());
//    cv::Mat imgdest = cv::Mat::zeros(img.size(), img.type());
//
//    // Apply horizontal pass
//    int totalRows = img.rows;
//#pragma omp parallel
//    {
//        int threadID = omp_get_thread_num();
//        int numThreads = omp_get_num_threads();
//        int rowsPerThread = totalRows / numThreads;
//        int startRow = threadID * rowsPerThread;
//        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;
//
//        cv::Mat subImg = img.rowRange(startRow, endRow);
//        cv::Mat subDest = applyLowPassFilter(subImg, horizontalKernel);
//        subDest.copyTo(tempImg.rowRange(startRow, endRow));
//    }
//
//    // Apply vertical pass
//#pragma omp parallel
//    {
//        int threadID = omp_get_thread_num();
//        int numThreads = omp_get_num_threads();
//        int rowsPerThread = totalRows / numThreads;
//        int startRow = threadID * rowsPerThread;
//        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;
//
//        int pad = kernelLength / 2;
//        int safeStart = std::max(startRow - pad, 0);
//        int safeEnd = std::min(endRow + pad, totalRows);
//
//        cv::Mat subImg = tempImg.rowRange(safeStart, safeEnd);
//        cv::Mat subDest = applyLowPassFilter(subImg, verticalKernel);
//
//        int copyStart = startRow - safeStart;
//        int copyLength = endRow - startRow;
//        subDest.rowRange(copyStart, copyStart + copyLength).copyTo(imgdest.rowRange(startRow, endRow));
//    }
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    // Display results
//    cv::imshow("Image Before", img);
//    cv::imshow("Image After (Parallel Filter)", imgdest);
//    cv::moveWindow("Image Before", 0, 45);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//    int channels = src.channels();
//
//    bool isHorizontal = (kernel.rows == 1);
//    bool isVertical = (kernel.cols == 1);
//
//    int padH = isVertical ? kernel.rows / 2 : 0;
//    int padW = isHorizontal ? kernel.cols / 2 : 0;
//
//    cv::Mat padded;
//    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);
//
//    std::vector<float> kernelValues;
//    if (isHorizontal) {
//        kernelValues.resize(kernel.cols);
//        for (int kx = 0; kx < kernel.cols; ++kx)
//            kernelValues[kx] = kernel.at<float>(0, kx);
//    } else if (isVertical) {
//        kernelValues.resize(kernel.rows);
//        for (int ky = 0; ky < kernel.rows; ++ky)
//            kernelValues[ky] = kernel.at<float>(ky, 0);
//    }
//
//    if (channels == 1) {
//        if (isHorizontal) {
//            for (int y = 0; y < src.rows; ++y) {
//                const uchar* paddedRow = padded.ptr<uchar>(y);
//                uchar* dstRow = dst.ptr<uchar>(y);
//#pragma omp simd
//                for (int x = 0; x < src.cols; ++x) {
//                    float sum = 0.0f;
//                    for (int kx = 0; kx < kernel.cols; ++kx)
//                        sum += paddedRow[x + kx] * kernelValues[kx];
//                    dstRow[x] = cv::saturate_cast<uchar>(sum);
//                }
//            }
//        } else if (isVertical) {
//            for (int y = 0; y < src.rows; ++y) {
//                uchar* dstRow = dst.ptr<uchar>(y);
//#pragma omp simd
//                for (int x = 0; x < src.cols; ++x) {
//                    float sum = 0.0f;
//                    for (int ky = 0; ky < kernel.rows; ++ky)
//                        sum += padded.ptr<uchar>(y + ky)[x] * kernelValues[ky];
//                    dstRow[x] = cv::saturate_cast<uchar>(sum);
//                }
//            }
//        }
//    } else {
//        if (isHorizontal) {
//            for (int y = 0; y < src.rows; ++y) {
//                const cv::Vec3b* paddedRow = padded.ptr<cv::Vec3b>(y);
//                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
//#pragma omp simd
//                for (int x = 0; x < src.cols; ++x) {
//                    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
//                    for (int kx = 0; kx < kernel.cols; ++kx) {
//                        const cv::Vec3b& pixel = paddedRow[x + kx];
//                        float kv = kernelValues[kx];
//                        sumB += pixel[0] * kv;
//                        sumG += pixel[1] * kv;
//                        sumR += pixel[2] * kv;
//                    }
//                    dstRow[x] = cv::Vec3b(
//                            cv::saturate_cast<uchar>(sumB),
//                            cv::saturate_cast<uchar>(sumG),
//                            cv::saturate_cast<uchar>(sumR)
//                    );
//                }
//            }
//        } else if (isVertical) {
//            for (int y = 0; y < src.rows; ++y) {
//                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
//#pragma omp simd
//                for (int x = 0; x < src.cols; ++x) {
//                    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
//                    for (int ky = 0; ky < kernel.rows; ++ky) {
//                        const cv::Vec3b& pixel = padded.ptr<cv::Vec3b>(y + ky)[x];
//                        float kv = kernelValues[ky];
//                        sumB += pixel[0] * kv;
//                        sumG += pixel[1] * kv;
//                        sumR += pixel[2] * kv;
//                    }
//                    dstRow[x] = cv::Vec3b(
//                            cv::saturate_cast<uchar>(sumB),
//                            cv::saturate_cast<uchar>(sumG),
//                            cv::saturate_cast<uchar>(sumR)
//                    );
//                }
//            }
//        }
//    }
//
//    return dst;
//}

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <omp.h>
//
//struct Image {
//    std::vector<std::vector<std::vector<unsigned char>>> data; // 3D for RGB or 2D for grayscale
//    int rows, cols, channels;
//};
//
//std::string getImagePath();
//Image loadImage(const std::string& path);
//Image convertToGrayscale(const Image& src);
//Image applyLowPassFilter(const Image& src, const std::vector<float>& kernel, bool isHorizontal);
//void displayImage(const Image& img, const std::string& windowName);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//
//    // Get valid image path
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    // Load image using OpenCV
//    Image img = loadImage(path);
//    if (img.rows == 0 || img.cols == 0) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    // Option to convert to grayscale
//    std::cout << "Do you want the image in grayscale? [y/n]: ";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        img = convertToGrayscale(img);
//    }
//
//    int kernelLength, kernelWidth;
//
//    // Get kernel dimensions
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//
//    std::cout << "\nYour configurations are:\n"
//              << "Image path: " << path << "\n"
//              << "Length = " << kernelLength << "\n"
//              << "Width = " << kernelWidth << "\n";
//
//    double start_time = omp_get_wtime();
//
//    // Create separable 1D kernels, normalized to sum to 1
//    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
//    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
//
//    // Apply horizontal pass
//    Image tempImg = applyLowPassFilter(img, horizontalKernel, true);
//
//    // Apply vertical pass
//    Image imgDest = applyLowPassFilter(tempImg, verticalKernel, false);
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    // Display results
//    displayImage(img, "Image Before");
//    displayImage(imgDest, "Image After (Parallel Filter)");
//    cv::moveWindow("Image Before", 0, 45);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
//Image loadImage(const std::string& path) {
//    cv::Mat cvImg = cv::imread(path, cv::IMREAD_UNCHANGED);
//    Image img;
//    if (cvImg.empty()) {
//        img.rows = 0;
//        img.cols = 0;
//        img.channels = 0;
//        return img;
//    }
//
//    img.rows = cvImg.rows;
//    img.cols = cvImg.cols;
//    img.channels = cvImg.channels();
//    img.data.resize(img.rows, std::vector<std::vector<unsigned char>>(img.cols, std::vector<unsigned char>(img.channels)));
//
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            if (img.channels == 1) {
//                img.data[y][x][0] = cvImg.at<unsigned char>(y, x);
//            } else {
//                cv::Vec3b pixel = cvImg.at<cv::Vec3b>(y, x);
//                img.data[y][x][0] = pixel[0]; // B
//                img.data[y][x][1] = pixel[1]; // G
//                img.data[y][x][2] = pixel[2]; // R
//            }
//        }
//    }
//    return img;
//}
//
//Image convertToGrayscale(const Image& src) {
//    Image dst;
//    dst.rows = src.rows;
//    dst.cols = src.cols;
//    dst.channels = 1;
//    dst.data.resize(dst.rows, std::vector<std::vector<unsigned char>>(dst.cols, std::vector<unsigned char>(1)));
//
//    for (int y = 0; y < src.rows; ++y) {
//        for (int x = 0; x < src.cols; ++x) {
//            if (src.channels == 1) {
//                dst.data[y][x][0] = src.data[y][x][0];
//            } else {
//                // Weighted grayscale: 0.299R + 0.587G + 0.114B
//                float gray = 0.299f * src.data[y][x][2] + 0.587f * src.data[y][x][1] + 0.114f * src.data[y][x][0];
//                dst.data[y][x][0] = static_cast<unsigned char>(gray);
//            }
//        }
//    }
//    return dst;
//}
//
//Image applyLowPassFilter(const Image& src, const std::vector<float>& kernel, bool isHorizontal) {
//    Image dst;
//    dst.rows = src.rows;
//    dst.cols = src.cols;
//    dst.channels = src.channels;
//    dst.data.resize(dst.rows, std::vector<std::vector<unsigned char>>(dst.cols, std::vector<unsigned char>(dst.channels)));
//
//    int kernelSize = kernel.size();
//    int pad = kernelSize / 2;
//
//    // Create padded image
//    Image padded;
//    padded.rows = src.rows + 2 * pad;
//    padded.cols = src.cols + 2 * pad;
//    padded.channels = src.channels;
//    padded.data.resize(padded.rows, std::vector<std::vector<unsigned char>>(padded.cols, std::vector<unsigned char>(padded.channels)));
//
//    // Replicate border
//    for (int y = 0; y < padded.rows; ++y) {
//        for (int x = 0; x < padded.cols; ++x) {
//            int srcY = std::min(std::max(y - pad, 0), src.rows - 1);
//            int srcX = std::min(std::max(x - pad, 0), src.cols - 1);
//            for (int c = 0; c < src.channels; ++c) {
//                padded.data[y][x][c] = src.data[srcY][srcX][c];
//            }
//        }
//    }
//
//    // Apply filter
//    int totalRows = src.rows;
//#pragma omp parallel
//    {
//        int threadID = omp_get_thread_num();
//        int numThreads = omp_get_num_threads();
//        int rowsPerThread = totalRows / numThreads;
//        int startRow = threadID * rowsPerThread;
//        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;
//
//        for (int y = startRow; y < endRow; ++y) {
//            for (int x = 0; x < src.cols; ++x) {
//                for (int c = 0; c < src.channels; ++c) {
//                    float sum = 0.0f;
//                    if (isHorizontal) {
//                        for (int kx = 0; kx < kernelSize; ++kx) {
//                            sum += padded.data[y + pad][x + kx][c] * kernel[kx];
//                        }
//                    } else {
//                        for (int ky = 0; ky < kernelSize; ++ky) {
//                            sum += padded.data[y + ky][x + pad][c] * kernel[ky];
//                        }
//                    }
//                    dst.data[y][x][c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//                }
//            }
//        }
//    }
//
//    return dst;
//}
//
//void displayImage(const Image& img, const std::string& windowName) {
//    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            if (img.channels == 1) {
//                cvImg.at<unsigned char>(y, x) = img.data[y][x][0];
//            } else {
//                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.data[y][x][0], img.data[y][x][1], img.data[y][x][2]);
//            }
//        }
//    }
//    cv::imshow(windowName, cvImg);
//}

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <omp.h>
//
//struct Image {
//    std::vector<unsigned char> data;
//    int rows, cols, channels;
//    Image(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r * c * ch) {}
//    unsigned char& at(int y, int x, int c) {
//        return data[(y * cols + x) * channels + c];
//    }
//    const unsigned char& at(int y, int x, int c) const {
//        return data[(y * cols + x) * channels + c];
//    }
//};
//
//std::string getImagePath();
//Image loadImage(const std::string& path);
//Image convertToGrayscale(const Image& src);
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal);
//void displayImage(const Image& img, const std::string& windowName);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    Image img = loadImage(path);
//    if (img.rows == 0 || img.cols == 0) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    std::cout << "Do you want the image in grayscale? [y/n]: ";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        img = convertToGrayscale(img);
//    }
//
//    int kernelLength, kernelWidth;
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//
//    std::cout << "\nYour configurations are:\n"
//              << "Image path: " << path << "\n"
//              << "Length = " << kernelLength << "\n"
//              << "Width = " << kernelWidth << "\n";
//
//    double start_time = omp_get_wtime();
//
//    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
//    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
//
//    // Apply filters in-place
//    applyLowPassFilter(img, horizontalKernel, true);
//    applyLowPassFilter(img, verticalKernel, false);
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    displayImage(img, "Image After (Parallel Filter)");
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
//Image loadImage(const std::string& path) {
//    cv::Mat cvImg = cv::imread(path, cv::IMREAD_UNCHANGED);
//    if (cvImg.empty()) {
//        return Image(0, 0, 0);
//    }
//    int rows = cvImg.rows;
//    int cols = cvImg.cols;
//    int channels = cvImg.channels();
//    Image img(rows, cols, channels);
//    if (channels == 1) {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
//            }
//        }
//    } else {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                cv::Vec3b pixel = cvImg.at<cv::Vec3b>(y, x);
//                img.at(y, x, 0) = pixel[0]; // B
//                img.at(y, x, 1) = pixel[1]; // G
//                img.at(y, x, 2) = pixel[2]; // R
//            }
//        }
//    }
//    return img;
//}
//
//Image convertToGrayscale(const Image& src) {
//    Image dst(src.rows, src.cols, 1);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < src.rows; ++y) {
//        for (int x = 0; x < src.cols; ++x) {
//            if (src.channels == 1) {
//                dst.at(y, x, 0) = src.at(y, x, 0);
//            } else {
//                float gray = 0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);
//                dst.at(y, x, 0) = static_cast<unsigned char>(gray);
//            }
//        }
//    }
//    return dst;
//}
//
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
//    int kernelSize = kernel.size();
//    int pad = kernelSize / 2;
//    int rows_padded = img.rows + 2 * pad;
//    int cols_padded = img.cols + 2 * pad;
//    std::vector<unsigned char> padded_data(rows_padded * cols_padded * img.channels);
//    std::vector<unsigned char> temp_data(img.data.size());
//
//    // Copy original data to padded array
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[((y + pad) * cols_padded + (x + pad)) * img.channels + c] = img.at(y, x, c);
//            }
//        }
//    }
//
//    // Top and bottom padding
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < pad; ++y) {
//        for (int x = 0; x < cols_padded; ++x) {
//            int src_x = std::min(std::max(x - pad, 0), img.cols - 1);
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[(y * cols_padded + x) * img.channels + c] = img.at(0, src_x, c);
//                padded_data[((rows_padded - 1 - y) * cols_padded + x) * img.channels + c] = img.at(img.rows - 1, src_x, c);
//            }
//        }
//    }
//
//    // Left and right padding
//#pragma omp parallel for collapse(2)
//    for (int y = pad; y < rows_padded - pad; ++y) {
//        for (int x = 0; x < pad; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[(y * cols_padded + x) * img.channels + c] = img.at(y - pad, 0, c);
//                padded_data[(y * cols_padded + (cols_padded - pad + x)) * img.channels + c] = img.at(y - pad, img.cols - 1, c);
//            }
//        }
//    }
//
//    // Apply filter
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//#pragma omp simd
//        for (int x = 0; x < img.cols; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                float sum = 0.0f;
//                if (isHorizontal) {
//                    for (int kx = 0; kx < kernelSize; ++kx) {
//                        int idx = ((y + pad) * cols_padded + (x + kx)) * img.channels + c;
//                        sum += static_cast<float>(padded_data[idx]) * kernel[kx];
//                    }
//                } else {
//                    for (int ky = 0; ky < kernelSize; ++ky) {
//                        int idx = ((y + ky) * cols_padded + (x + pad)) * img.channels + c;
//                        sum += static_cast<float>(padded_data[idx]) * kernel[ky];
//                    }
//                }
//                temp_data[(y * img.cols + x) * img.channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//            }
//        }
//    }
//
//    // Copy back to img.data
//    img.data = std::move(temp_data);
//}
//
//void displayImage(const Image& img, const std::string& windowName) {
//    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            if (img.channels == 1) {
//                cvImg.at<unsigned char>(y, x) = img.at(y, x, 0);
//            } else {
//                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
//            }
//        }
//    }
//    cv::imshow(windowName, cvImg);
//}

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <omp.h>
//
//struct Image {
//    std::vector<unsigned char> data;
//    int rows, cols, channels;
//    Image(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r * c * ch) {}
//    unsigned char& at(int y, int x, int c) {
//        return data[(y * cols + x) * channels + c];
//    }
//    const unsigned char& at(int y, int x, int c) const {
//        return data[(y * cols + x) * channels + c];
//    }
//};
//
//std::string getImagePath();
//Image loadImage(const std::string& path);
//Image convertToGrayscale(const Image& src);
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal);
//void displayImage(const Image& img, const std::string& windowName);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    Image img = loadImage(path);
//    if (img.rows == 0 || img.cols == 0) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    std::cout << "Do you want the image in grayscale? [y/n]: ";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        img = convertToGrayscale(img);
//    }
//
//    int kernelLength, kernelWidth;
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//
//    std::cout << "\nYour configurations are:\n"
//              << "Image path: " << path << "\n"
//              << "Length = " << kernelLength << "\n"
//              << "Width = " << kernelWidth << "\n";
//
//    double start_time = omp_get_wtime();
//
//    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
//    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
//
//    applyLowPassFilter(img, horizontalKernel, true);
//    applyLowPassFilter(img, verticalKernel, false);
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    displayImage(img, "Image After (Parallel Filter)");
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
//Image loadImage(const std::string& path) {
//    cv::Mat cvImg = cv::imread(path, cv::IMREAD_UNCHANGED);
//    if (cvImg.empty()) {
//        return Image(0, 0, 0);
//    }
//    int rows = cvImg.rows;
//    int cols = cvImg.cols;
//    int channels = cvImg.channels();
//    Image img(rows, cols, channels);
//    if (channels == 1) {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
//            }
//        }
//    } else {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                cv::Vec3b pixel = cvImg.at<cv::Vec3b>(y, x);
//                img.at(y, x, 0) = pixel[0]; // B
//                img.at(y, x, 1) = pixel[1]; // G
//                img.at(y, x, 2) = pixel[2]; // R
//            }
//        }
//    }
//    return img;
//}
//
//Image convertToGrayscale(const Image& src) {
//    Image dst(src.rows, src.cols, 1);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < src.rows; ++y) {
//        for (int x = 0; x < src.cols; ++x) {
//            if (src.channels == 1) {
//                dst.at(y, x, 0) = src.at(y, x, 0);
//            } else {
//                float gray = 0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);
//                dst.at(y, x, 0) = static_cast<unsigned char>(gray);
//            }
//        }
//    }
//    return dst;
//}
//
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
//    int kernelSize = kernel.size();
//    int pad = kernelSize / 2;
//    int rows_padded = img.rows + 2 * pad;
//    int cols_padded = img.cols + 2 * pad;
//    std::vector<unsigned char> padded_data(rows_padded * cols_padded * img.channels);
//    std::vector<unsigned char> temp_data(img.data.size());
//
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[((y + pad) * cols_padded + (x + pad)) * img.channels + c] = img.at(y, x, c);
//            }
//        }
//    }
//
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < pad; ++y) {
//        for (int x = 0; x < cols_padded; ++x) {
//            int src_x = std::min(std::max(x - pad, 0), img.cols - 1);
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[(y * cols_padded + x) * img.channels + c] = img.at(0, src_x, c);
//                padded_data[((rows_padded - 1 - y) * cols_padded + x) * img.channels + c] = img.at(img.rows - 1, src_x, c);
//            }
//        }
//    }
//
//#pragma omp parallel for collapse(2)
//    for (int y = pad; y < rows_padded - pad; ++y) {
//        for (int x = 0; x < pad; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                padded_data[(y * cols_padded + x) * img.channels + c] = img.at(y - pad, 0, c);
//                padded_data[(y * cols_padded + (cols_padded - 1 - x)) * img.channels + c] = img.at(y - pad, img.cols - 1, c);
//            }
//        }
//    }
//
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            for (int c = 0; c < img.channels; ++c) {
//                float sum = 0.0f;
//                if (isHorizontal) {
//                    for (int kx = 0; kx < kernelSize; ++kx) {
//                        int idx = ((y + pad) * cols_padded + (x + kx)) * img.channels + c;
//                        sum += static_cast<float>(padded_data[idx]) * kernel[kx];
//                    }
//                } else {
//                    for (int ky = 0; ky < kernelSize; ++ky) {
//                        int idx = ((y + ky) * cols_padded + (x + pad)) * img.channels + c;
//                        sum += static_cast<float>(padded_data[idx]) * kernel[ky];
//                    }
//                }
//                temp_data[(y * img.cols + x) * img.channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//            }
//        }
//    }
//
//    img.data = std::move(temp_data);
//}
//
//void displayImage(const Image& img, const std::string& windowName) {
//    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            if (img.channels == 1) {
//                cvImg.at<unsigned char>(y, x) = img.at(y, x, 0);
//            } else {
//                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
//            }
//        }
//    }
//    cv::imshow(windowName, cvImg);
//}

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <omp.h>
//
//struct Image {
//    std::vector<unsigned char> data;
//    int rows, cols, channels;
//    Image(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r * c * ch) {}
//    unsigned char& at(int y, int x, int c) {
//        return data[(y * cols + x) * channels + c];
//    }
//    const unsigned char& at(int y, int x, int c) const {
//        return data[(y * cols + x) * channels + c];
//    }
//};
//
//std::string getImagePath();
//Image loadImage(const std::string& path);
//Image convertToGrayscale(const Image& src);
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal);
//void displayImage(const Image& img, const std::string& windowName);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    Image img = loadImage(path);
//    if (img.rows == 0 || img.cols == 0) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    std::cout << "Do you want the image in grayscale? [y/n]: ";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        img = convertToGrayscale(img);
//    }
//
//    int kernelLength, kernelWidth;
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//
//    std::cout << "\nYour configurations are:\n"
//              << "Image path: " << path << "\n"
//              << "Length = " << kernelLength << "\n"
//              << "Width = " << kernelWidth << "\n";
//
//    double start_time = omp_get_wtime();
//
//    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
//    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
//
//    applyLowPassFilter(img, horizontalKernel, true);
//    applyLowPassFilter(img, verticalKernel, false);
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    displayImage(img, "Image After (Parallel Filter)");
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
//Image loadImage(const std::string& path) {
//    cv::Mat cvImg = cv::imread(path, cv::IMREAD_UNCHANGED);
//    if (cvImg.empty()) {
//        return Image(0, 0, 0);
//    }
//    int rows = cvImg.rows;
//    int cols = cvImg.cols;
//    int channels = cvImg.channels();
//    Image img(rows, cols, channels);
//    if (channels == 1) {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
//            }
//        }
//    } else {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                cv::Vec3b pixel = cvImg.at<cv::Vec3b>(y, x);
//                img.at(y, x, 0) = pixel[0]; // B
//                img.at(y, x, 1) = pixel[1]; // G
//                img.at(y, x, 2) = pixel[2]; // R
//            }
//        }
//    }
//    return img;
//}
//
//Image convertToGrayscale(const Image& src) {
//    Image dst(src.rows, src.cols, 1);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < src.rows; ++y) {
//        for (int x = 0; x < src.cols; ++x) {
//            if (src.channels == 1) {
//                dst.at(y, x, 0) = src.at(y, x, 0);
//            } else {
//                float gray = 0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);
//                dst.at(y, x, 0) = static_cast<unsigned char>(gray);
//            }
//        }
//    }
//    return dst;
//}
//
//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
//    int kernelSize = kernel.size();
//    int pad = kernelSize / 2;
//    int rows = img.rows;
//    int cols = img.cols;
//    int channels = img.channels;
//    std::vector<unsigned char> temp_data(rows * cols * channels);
//
//    if (isHorizontal) {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    float sum = 0.0f;
//                    for (int kx = -pad; kx <= pad; ++kx) {
//                        int nx = std::max(0, std::min(x + kx, cols - 1));
//                        sum += static_cast<float>(img.at(y, nx, c)) * kernel[kx + pad];
//                    }
//                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//                }
//            }
//        }
//    } else {
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < rows; ++y) {
//            for (int x = 0; x < cols; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    float sum = 0.0f;
//                    for (int ky = -pad; ky <= pad; ++ky) {
//                        int ny = std::max(0, std::min(y + ky, rows - 1));
//                        sum += static_cast<float>(img.at(ny, x, c)) * kernel[ky + pad];
//                    }
//                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//                }
//            }
//        }
//    }
//
//    img.data = std::move(temp_data);
//}
//
//void displayImage(const Image& img, const std::string& windowName) {
//    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
//#pragma omp parallel for collapse(2)
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            if (img.channels == 1) {
//                cvImg.at<unsigned char>(y, x) = img.at(y, x, 0);
//            } else {
//                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
//            }
//        }
//    }
//    cv::imshow(windowName, cvImg);
//}

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <opencv2/imgproc.hpp>

struct Image {
    std::vector<unsigned char> data;
    int rows, cols, channels;
    Image(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r * c * ch) {}
    unsigned char& at(int y, int x, int c) {
        return data[(y * cols + x) * channels + c];
    }
    const unsigned char& at(int y, int x, int c) const {
        return data[(y * cols + x) * channels + c];
    }
};

std::string getImagePath();
Image loadImage(const std::string& path);
Image convertToGrayscale(const Image& src);
void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal);
void displayImage(const Image& img, const std::string& windowName);
cv::Mat imageToMat(const Image& img);

int main() {
    std::string path;
    bool flag = false;
    std::string grayFlag = "y";

    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));

    Image img = loadImage(path);
    cv::Mat cvImgBefore= imageToMat(img);
    if (img.rows == 0 || img.cols == 0) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    std::cout << "Do you want the image in grayscale? [y/n]: ";
    std::cin >> grayFlag;

    if (grayFlag[0] == 'y') {
        img = convertToGrayscale(img);
    }

    int kernelLength, kernelWidth;
    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;

    std::cout << "\nYour configurations are:\n"
              << "Image path: " << path << "\n"
              << "Length = " << kernelLength << "\n"
              << "Width = " << kernelWidth << "\n";

    double start_time = omp_get_wtime();

    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
    applyLowPassFilter(img, horizontalKernel, true);
    applyLowPassFilter(img, verticalKernel, false);

    double end_time = omp_get_wtime();
    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
    displayImage(img, "Image After (Parallel Filter)");
    cv::Mat cvImgAfter = imageToMat(img);
    bool success = cv::imwrite("output.jpg", cvImgAfter);
    if (!success) {
        std::cerr << "Failed to save image!" << std::endl;
        return -1;
    }
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

Image loadImage(const std::string& path) {
    cv::Mat cvImg = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (cvImg.empty()) {
        return Image(0, 0, 0);
    }
    int rows = cvImg.rows;
    int cols = cvImg.cols;
    int channels = cvImg.channels();
    Image img(rows, cols, channels);
    if (channels == 1) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
            }
        }
    } else {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                cv::Vec3b pixel = cvImg.at<cv::Vec3b>(y, x);
                img.at(y, x, 0) = pixel[0]; // B
                img.at(y, x, 1) = pixel[1]; // G
                img.at(y, x, 2) = pixel[2]; // R
            }
        }
    }
    return img;
}

Image convertToGrayscale(const Image& src) {
    Image dst(src.rows, src.cols, 1);
#pragma omp parallel for collapse(2)
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            if (src.channels == 1) {
                dst.at(y, x, 0) = src.at(y, x, 0);
            } else {
                float gray = 0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);
                dst.at(y, x, 0) = static_cast<unsigned char>(gray);
            }
        }
    }
    return dst;
}

//void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
//    int kernelSize = kernel.size();
//    int pad = kernelSize / 2;
//    int rows = img.rows;
//    int cols = img.cols;
//    int channels = img.channels;
//    std::vector<unsigned char> temp_data(rows * cols * channels);
//
//    if (isHorizontal) {
//#pragma omp parallel for
//        for (int y = 0; y < rows; ++y) {
//            std::vector<float> padded_row((cols + 2 * pad) * channels);
//            // Precompute padded row
//            for (int x = 0; x < cols; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_row[(x + pad) * channels + c] = img.at(y, x, c);
//                }
//            }
//            // Pad left
//            for (int x = 0; x < pad; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_row[x * channels + c] = img.at(y, 0, c);
//                }
//            }
//            // Pad right
//            for (int x = cols + pad; x < cols + 2 * pad; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_row[x * channels + c] = img.at(y, cols - 1, c);
//                }
//            }
//            // Apply filter without boundary checks
//            for (int x = 0; x < cols; ++x) {
//                for (int c = 0; c < channels; ++c) {
//                    float sum = 0.0f;
//                    for (int kx = 0; kx < kernelSize; ++kx) {
//                        sum += padded_row[(x + kx) * channels + c] * kernel[kx];
//                    }
//                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//                }
//            }
//        }
//    } else {
//#pragma omp parallel for
//        for (int x = 0; x < cols; ++x) {
//            std::vector<float> padded_col((rows + 2 * pad) * channels);
//            // Precompute padded column
//            for (int y = 0; y < rows; ++y) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_col[(y + pad) * channels + c] = img.at(y, x, c);
//                }
//            }
//            // Pad top
//            for (int y = 0; y < pad; ++y) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_col[y * channels + c] = img.at(0, x, c);
//                }
//            }
//            // Pad bottom
//            for (int y = rows + pad; y < rows + 2 * pad; ++y) {
//                for (int c = 0; c < channels; ++c) {
//                    padded_col[y * channels + c] = img.at(rows - 1, x, c);
//                }
//            }
//            // Apply filter without boundary checks
//            for (int y = 0; y < rows; ++y) {
//                for (int c = 0; c < channels; ++c) {
//                    float sum = 0.0f;
//                    for (int ky = 0; ky < kernelSize; ++ky) {
//                        sum += padded_col[(y + ky) * channels + c] * kernel[ky];
//                    }
//                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
//                }
//            }
//        }
//    }
//
//    img.data = std::move(temp_data);
//}

void displayImage(const Image& img, const std::string& windowName) {
    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
#pragma omp parallel for collapse(2)
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.channels == 1) {
                cvImg.at<unsigned char>(y, x) = img.at(y, x, 0);
            } else {
                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
            }
        }
    }
    cv::imshow(windowName, cvImg);
}

void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
    int kernelSize = kernel.size();
    int pad = kernelSize / 2;
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels;
    std::vector<unsigned char> temp_data(rows * cols * channels);
    float factor = 1.0f / kernelSize;

    if (isHorizontal) {
#pragma omp parallel for
        for (int y = 0; y < rows; ++y) {
            std::vector<unsigned char> padded_row((cols + 2 * pad) * channels);
            // Fill padded_row with boundary replication
            for (int x = 0; x < pad; ++x) {
                for (int c = 0; c < channels; ++c) {
                    padded_row[x * channels + c] = img.at(y, 0, c);
                }
            }
            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    padded_row[(x + pad) * channels + c] = img.at(y, x, c);
                }
            }
            for (int x = cols + pad; x < cols + 2 * pad; ++x) {
                for (int c = 0; c < channels; ++c) {
                    padded_row[x * channels + c] = img.at(y, cols - 1, c);
                }
            }
            // Initialize sums for the first window
            std::vector<float> sums(channels, 0.0f);
            for (int kx = 0; kx < kernelSize; ++kx) {
                for (int c = 0; c < channels; ++c) {
                    sums[c] += static_cast<float>(padded_row[kx * channels + c]);
                }
            }
            // Apply sliding window sum
            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    float filtered_value = sums[c] * factor;
                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(
                            std::min(std::max(filtered_value, 0.0f), 255.0f)
                    );
                }
                if (x < cols - 1) {
                    for (int c = 0; c < channels; ++c) {
                        sums[c] -= static_cast<float>(padded_row[x * channels + c]);
                        sums[c] += static_cast<float>(padded_row[(x + kernelSize) * channels + c]);
                    }
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int x = 0; x < cols; ++x) {
            std::vector<unsigned char> padded_col((rows + 2 * pad) * channels);
            // Fill padded_col with boundary replication
            for (int y = 0; y < pad; ++y) {
                for (int c = 0; c < channels; ++c) {
                    padded_col[y * channels + c] = img.at(0, x, c);
                }
            }
            for (int y = 0; y < rows; ++y) {
                for (int c = 0; c < channels; ++c) {
                    padded_col[(y + pad) * channels + c] = img.at(y, x, c);
                }
            }
            for (int y = rows + pad; y < rows + 2 * pad; ++y) {
                for (int c = 0; c < channels; ++c) {
                    padded_col[y * channels + c] = img.at(rows - 1, x, c);
                }
            }
            // Initialize sums for the first window
            std::vector<float> sums(channels, 0.0f);
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int c = 0; c < channels; ++c) {
                    sums[c] += static_cast<float>(padded_col[ky * channels + c]);
                }
            }
            // Apply sliding window sum
            for (int y = 0; y < rows; ++y) {
                for (int c = 0; c < channels; ++c) {
                    float filtered_value = sums[c] * factor;
                    temp_data[(y * cols + x) * channels + c] = static_cast<unsigned char>(
                            std::min(std::max(filtered_value, 0.0f), 255.0f)
                    );
                }
                if (y < rows - 1) {
                    for (int c = 0; c < channels; ++c) {
                        sums[c] -= static_cast<float>(padded_col[y * channels + c]);
                        sums[c] += static_cast<float>(padded_col[(y + kernelSize) * channels + c]);
                    }
                }
            }
        }
    }

    img.data = std::move(temp_data);
}

cv::Mat imageToMat(const Image& img) {
    cv::Mat mat(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.channels == 1) {
                mat.at<unsigned char>(y, x) = img.at(y, x, 0);
            } else {
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
            }
        }
    }
    return mat;
}

Image matToImage(const cv::Mat& mat) {
    int channels = mat.channels();
    Image img(mat.rows, mat.cols, channels);
    if (channels == 1) {
        for (int y = 0; y < mat.rows; ++y) {
            for (int x = 0; x < mat.cols; ++x) {
                img.at(y, x, 0) = mat.at<unsigned char>(y, x);
            }
        }
    } else {
        for (int y = 0; y < mat.rows; ++y) {
            for (int x = 0; x < mat.cols; ++x) {
                cv::Vec3b pixel = mat.at<cv::Vec3b>(y, x);
                img.at(y, x, 0) = pixel[0];
                img.at(y, x, 1) = pixel[1];
                img.at(y, x, 2) = pixel[2];
            }
        }
    }
    return img;
}
