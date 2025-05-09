#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <chrono>

struct Image {
    std::vector<unsigned char> data;
    int rows, cols, channels;
    Image(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r* c* ch) {}
    unsigned char& at(int y, int x, int c) {
        return data[(y * cols + x) * channels + c];
    }
    const unsigned char& at(int y, int x, int c) const {
        return data[(y * cols + x) * channels + c];
    }
};

int userThreads = 1;

std::string getImagePath();
Image loadImage(const std::string& path);
Image convertToGrayscale(const Image& src);
void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal);
void displayImage(const Image& img, const std::string& windowName);
cv::Mat imageToMat(const Image& img);
int getValidThreadCount();

int main() {
    std::string path;
    bool flag = false;
    std::string grayFlag = "y";

    userThreads = getValidThreadCount();

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));

    Image img = loadImage(path);
    cv::Mat cvImgBefore = imageToMat(img);
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


    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> horizontalKernel(kernelWidth, 1.0f / kernelWidth);
    std::vector<float> verticalKernel(kernelLength, 1.0f / kernelLength);
    applyLowPassFilter(img, horizontalKernel, true);
    applyLowPassFilter(img, verticalKernel, false);



    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_time = duration.count();

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Statistics:" << std::endl;
    std::cout << "Total execution time: " << elapsed_time << " milliseconds" << std::endl;
    std::cout << "Image size: " << img.cols << " x " << img.rows << " pixels" << std::endl;
    std::cout << "Number of processes: " << userThreads << std::endl;
    std::cout << "Kernel size: " << kernelLength << " x " << kernelWidth << std::endl;
    std::cout << "----------------------------------------" << std::endl;

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
    cv::Mat cvImg = cv::imread(path, cv::IMREAD_COLOR);
    if (cvImg.empty()) {
        return Image(0, 0, 0);
    }
    int rows = cvImg.rows;
    int cols = cvImg.cols;
    int channels = cvImg.channels();
    Image img(rows, cols, channels);
    if (channels == 1) {
#pragma omp parallel for num_threads(userThreads) collapse(2)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
            }
        }
    }
    else {
#pragma omp parallel for num_threads(userThreads) collapse(2)
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
#pragma omp parallel for num_threads(userThreads) collapse(2)
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            if (src.channels == 1) {
                dst.at(y, x, 0) = src.at(y, x, 0);
            }
            else {
                float gray = 0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);
                dst.at(y, x, 0) = static_cast<unsigned char>(gray);
            }
        }
    }
    return dst;
}

void displayImage(const Image& img, const std::string& windowName) {
    cv::Mat cvImg(img.rows, img.cols, img.channels == 1 ? CV_8UC1 : CV_8UC3);
#pragma omp parallel for num_threads(userThreads) collapse(2)
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.channels == 1) {
                cvImg.at<unsigned char>(y, x) = img.at(y, x, 0);
            }
            else {
                cvImg.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
            }
        }
    }
    cv::imshow(windowName, cvImg);
}

std::vector<unsigned char> createPaddedRow(const Image& img, int y, int pad, int channels) {
    int cols = img.cols;
    std::vector<unsigned char> padded_row((cols + 2 * pad) * channels);
    // Left padding
    for (int x = 0; x < pad; ++x) {
        for (int c = 0; c < channels; ++c) {
            padded_row[x * channels + c] = img.at(y, 0, c);
        }
    }
    // Original row
    for (int x = 0; x < cols; ++x) {
        for (int c = 0; c < channels; ++c) {
            padded_row[(x + pad) * channels + c] = img.at(y, x, c);
        }
    }
    // Right padding
    for (int x = cols + pad; x < cols + 2 * pad; ++x) {
        for (int c = 0; c < channels; ++c) {
            padded_row[x * channels + c] = img.at(y, cols - 1, c);
        }
    }
    return padded_row;
}

std::vector<unsigned char> createPaddedCol(const Image& img, int x, int pad, int channels) {
    int rows = img.rows;
    std::vector<unsigned char> padded_col((rows + 2 * pad) * channels);
    // Top padding
    for (int y = 0; y < pad; ++y) {
        for (int c = 0; c < channels; ++c) {
            padded_col[y * channels + c] = img.at(0, x, c);
        }
    }
    // Original column
    for (int y = 0; y < rows; ++y) {
        for (int c = 0; c < channels; ++c) {
            padded_col[(y + pad) * channels + c] = img.at(y, x, c);
        }
    }
    // Bottom padding
    for (int y = rows + pad; y < rows + 2 * pad; ++y) {
        for (int c = 0; c < channels; ++c) {
            padded_col[y * channels + c] = img.at(rows - 1, x, c);
        }
    }
    return padded_col;
}

void filterRow(const std::vector<unsigned char>& padded_row, int cols, int channels,
               int kernelSize, float factor, std::vector<unsigned char>& temp_data, int y) {
    std::vector<float> sums(channels, 0.0f);
    // Initialize sums
    for (int kx = 0; kx < kernelSize; ++kx) {
        for (int c = 0; c < channels; ++c) {
            sums[c] += static_cast<float>(padded_row[kx * channels + c]);
        }
    }
    // Apply sliding window
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

void filterCol(const std::vector<unsigned char>& padded_col, int rows, int channels,
               int kernelSize, float factor, std::vector<unsigned char>& temp_data,
               int x, int cols) {
    std::vector<float> sums(channels, 0.0f);
    // Initialize sums
    for (int ky = 0; ky < kernelSize; ++ky) {
        for (int c = 0; c < channels; ++c) {
            sums[c] += static_cast<float>(padded_col[ky * channels + c]);
        }
    }
    // Apply sliding window
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

void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
    int kernelSize = kernel.size();
    int pad = kernelSize / 2;
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels;
    std::vector<unsigned char> temp_data(rows * cols * channels);
    float factor = 1.0f / kernelSize;

    if (isHorizontal) {
#pragma omp parallel for num_threads(userThreads)
        for (int y = 0; y < rows; ++y) {
            std::vector<unsigned char> padded_row = createPaddedRow(img, y, pad, channels);
            filterRow(padded_row, cols, channels, kernelSize, factor, temp_data, y);
        }
    } else {
#pragma omp parallel for num_threads(userThreads)
        for (int x = 0; x < cols; ++x) {
            std::vector<unsigned char> padded_col = createPaddedCol(img, x, pad, channels);
            filterCol(padded_col, rows, channels, kernelSize, factor, temp_data, x, cols);
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
            }
            else {
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
            }
        }
    }
    return mat;
}

int getValidThreadCount() {
    int userThreads = 1;
    bool validInput = false;

    std::cout << "To use the max processes of your computer enter (0)\n";
    std::cout << "Enter number of threads to use: ";

    while (!validInput) {
        if (std::cin >> userThreads) {
            if (userThreads < 0) {
                std::cout << "Error: Number of threads cannot be negative.\n";
                std::cout << "Please enter a positive number (or 0 for max): ";
            }
            else {
                validInput = true;
            }
        }
        else {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Error: Please enter a valid number.\n";
            std::cout << "Enter number of threads to use: ";
        }
    }

    if (userThreads == 0 || userThreads > omp_get_max_threads()) {
        userThreads = omp_get_max_threads();
        std::cout << "Using maximum available threads: " << userThreads << std::endl;
    }

    return userThreads;
}