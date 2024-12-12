#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main() {
    // Load class names
    vector<string> class_names = {"damaged", "healthy ripe", "unripe"};

    // Load ONNX model
    auto model = readNetFromONNX("models/best.onnx");
    if (model.empty()) {
        cerr << "Error: Could not load the ONNX model." << endl;
        return -1;
    }

    // Read and preprocess the input image
    Mat image = imread("tmp/tomato.jpg");
    if (image.empty()) {
        cerr << "Error: Could not read input image." << endl;
        return -1;
    }

    Mat blob = blobFromImage(image, 1.0 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    model.setInput(blob);

    // Perform forward pass
    Mat output = model.forward();

    // Reshape the output to [8400, 7]
    Mat detectionMat(output.size[2], output.size[1], CV_32F, output.ptr<float>());

    // Iterate through detections
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        int class_id = static_cast<int>(detectionMat.at<float>(i, 1));

        // Skip detections with low confidence or invalid class ID
        if (confidence < 0.4 || class_id < 0 || class_id >= class_names.size())
            continue;

        // Extract bounding box coordinates
        float x_min = detectionMat.at<float>(i, 3) * image.cols;
        float y_min = detectionMat.at<float>(i, 4) * image.rows;
        float x_max = detectionMat.at<float>(i, 5) * image.cols;
        float y_max = detectionMat.at<float>(i, 6) * image.rows;

        // Draw the bounding box
        rectangle(image, Point(static_cast<int>(x_min), static_cast<int>(y_min)),
                  Point(static_cast<int>(x_max), static_cast<int>(y_max)),
                  Scalar(255, 255, 255), 2);

        // Add label
        string label = format("%s: %.2f", class_names[class_id].c_str(), confidence);
        putText(image, label, Point(static_cast<int>(x_min), static_cast<int>(y_min) - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    }

    // Display the image
    imshow("Detections", image);
    imwrite("image_result.jpg", image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
