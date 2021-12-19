#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "iostream"
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

int coinDetect(string imgPath) {
    cv::Mat img = cv::imread(imgPath);

    // Grayscale
    cv::Mat graymat;
    cv::cvtColor(img, graymat, cv::COLOR_BGR2GRAY);

    // Gaussian smoothing
    cv::Mat gaussMat;
    cv::GaussianBlur(graymat, gaussMat, Size(17, 17), 0);

    // Canny edge detection
    cv::Mat cannyMat;
    cv::Canny(gaussMat, cannyMat, 30, 200, 3);

    // Binarization
    cv::Mat otsuMat;
    cv::threshold(cannyMat, otsuMat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Erosion
    cv::Mat dilaMat;
    int dilation_size = 1;
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(otsuMat,dilaMat, element);
    
    // Hough transform
    cv::Mat rel;
    cv::cvtColor(dilaMat, rel, cv::COLOR_GRAY2BGR);
    std::vector<Vec3f> circles;
    HoughCircles(dilaMat, circles, HOUGH_GRADIENT, 1,
        dilaMat.rows / 16,
        100, 30, 0, 100
    );
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(img, center, 1, Scalar(0, 255, 0), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(img, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
        cout << "x=" << center.x << " y=" << center.y << " radius:" << radius << "\n";
    }
    
    cout << "Number of coins: " << circles.size() << "\n";
    cv::imwrite("output.jpg", img);
    cv::imshow("result", img);
    cv::waitKey(0);
    return circles.size();
}

float calcF1Score(int numberCoins, string txtPath) {
    std::ifstream inFile(txtPath);
    float realNumberCoins = std::count(std::istreambuf_iterator<char>(inFile),
        std::istreambuf_iterator<char>(), '\n') + 1;
    float TP = numberCoins;
    float FP = 0;
    float FN = 0;
    if (realNumberCoins < numberCoins) {
        FP = numberCoins - realNumberCoins;
    }
    else {
        FN = realNumberCoins - numberCoins;
    }
    float Precision = TP / (TP + FP);
    float Recall = TP / (TP + FN);
    float F1 = 2 * Precision * Recall / (Precision + Recall);
    cout << "Precision:" << Precision << "\n";
    cout << "Recall:" << Recall << "\n";
    cout << "F1:" << F1 << "\n";
    return F1;
}

int main(int argc, char* argv[]) {
    std::string imgPath(argv[1]);
    // Detect coin in input.jpg
    coinDetect(imgPath);
    // Calculate F1 score of dataset, uncomment to run code
    /*
    string imgArr[15] = {
        "./dataset/01.jpg",
        "./dataset/02.jpg",
        "./dataset/03.jpg",
        "./dataset/04.jpg",
        "./dataset/05.jpg",
        "./dataset/06.jpg",
        "./dataset/07.jpg",
        "./dataset/08.jpg",
        "./dataset/09.jpg",
        "./dataset/10.jpg",
        "./dataset/11.jpg",
        "./dataset/12.jpg",
        "./dataset/13.jpg",
        "./dataset/14.jpg",
        "./dataset/15.jpg"
    };
    string txtArr[15] = {
        "./dataset/01.txt",
        "./dataset/02.txt",
        "./dataset/03.txt",
        "./dataset/04.txt",
        "./dataset/05.txt",
        "./dataset/06.txt",
        "./dataset/07.txt",
        "./dataset/08.txt",
        "./dataset/09.txt",
        "./dataset/10.txt",
        "./dataset/11.txt",
        "./dataset/12.txt",
        "./dataset/13.txt",
        "./dataset/14.txt",
        "./dataset/15.txt"
    };
    float F1[15];
    float F1Total = 0;
    for (int i = 0; i < 15; i++) {
        F1[i] = calcF1Score(coinDetect(imgArr[i]), txtArr[i]);
    }
    cout << "\n";
    for (int i = 0; i < 15; i++) {
        cout << F1[i] << "\t";
        F1Total += F1[i];
    }
    F1Total /= 15;
    cout << "\n" << F1Total;
    */
    return 0;
}