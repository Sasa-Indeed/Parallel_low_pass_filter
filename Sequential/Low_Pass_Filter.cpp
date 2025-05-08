#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/imgproc.hpp>
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
    std::cout << "Sequential execution" << std::endl;
    std::cout << "Kernel size: " << kernelLength << " x " << kernelWidth << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    displayImage(img, "Image After (Sequential Filter)");
    cv::Mat cvImgAfter = imageToMat(img);

    bool success = cv::imwrite("output_sequential.jpg", cvImgAfter);
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
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                img.at(y, x, 0) = cvImg.at<unsigned char>(y, x);
            }
        }
    }
    else {
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

void applyLowPassFilter(Image& img, const std::vector<float>& kernel, bool isHorizontal) {
    int kernelSize = kernel.size();
    int pad = kernelSize / 2;
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels;
    std::vector<unsigned char> temp_data(rows * cols * channels);
    float factor = 1.0f / kernelSize;

    if (isHorizontal) {
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
    }
    else {
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
            }
            else {
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
    }
    else {
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