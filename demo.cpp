#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "yolo.h"
#include "logging.h"
#include <ctime>
#include <csignal>
#include <cstring>
#include <string>

using std::vector;

static Logger gLogger;
static Logger gLogger2;

// 전역 변수로 자원을 관리
bool interrupted = false;

// 시그널 핸들러 함수
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    interrupted = true;
}

class Tester {
public:
    Tester(std::string modelPath, std::string yoloPath, std::string width, std::string height) {
        out.clear();
        DS = new DeepSort(modelPath, 128, 256, 0, &gLogger);
        std::cout << "DeepSort initialized!" << "\n";
        int oWidth = std::stoi(width);
        int oHeight = std::stoi(height);
        this->width = oWidth;
        this->height = oHeight;
        int mWidth = convertHW(oWidth);
        int mHeight = convertHW(oHeight);
        yolo = new Yolo(yoloPath, cv::Size(mWidth, mHeight), "classes.txt", true);
        std::cout << "Yolo initialized!" << "\n";
    }
    ~Tester() {
        delete DS;
        delete yolo;
        std::cout << "DeepSort released!" << "\n";
    }

public:

    void split(const std::string& s, vector<std::string>& token, char delim=' ') {
        token.clear();
        auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
            for (size_t i = pos; i < s.size(); ++i)
                if (s[i] != delim) return i;
            return std::string::npos;
        };
        size_t lastPos = string_find_first_not(0);
        size_t pos = s.find(delim, lastPos);
        while (lastPos != std::string::npos) {
            token.emplace_back(s.substr(lastPos, pos-lastPos));
            lastPos = string_find_first_not(pos);
            pos = s.find(delim, lastPos);
        }
    }

    void run(std::string videoPath) {
        std::cout << "run Called " << videoPath << "\n";
        // 마지막 '/' 또는 '\'의 위치 찾기
        size_t last_slash_pos = videoPath.find_last_of("/\\");
        
        // 마지막 슬래시 이후의 문자열 추출
        std::string filename = videoPath.substr(last_slash_pos + 1);
        cv::VideoCapture cap(videoPath); // Open the video file
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter videoWriter(filename, cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(width * scale, height * scale));
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file" << std::endl;
            return;
        } else {
            std::cout << "Video file opened successfully" << std::endl;
        }

        cv::Mat frame;
        int frameIndex = 1;
        vector<DetectBox> inBoxes;
        vector<DetectBox> outBoxes;
        vector<cv::Scalar> colors;
        vector<std::string> classNames;
        while (cap.read(frame)) {
            if (interrupted) {
                break;
            }
            
            inBoxes.clear();
            outBoxes.clear();
            colors.clear();
            std::vector<Detection> output = yolo->runYolo(frame);

            int detections = output.size();

            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];

                cv::Rect box = detection.box;
                cv::Scalar color = detection.color;
                colors.push_back(color);
                classNames.push_back(detection.className);

                int x1 = box.x;
                int y1 = box.y;
                int w = box.width;
                int h = box.height;
                int x2 = x1 + w;
                int y2 = y1 + h;
                int cls = detection.class_id;
                float conf = detection.confidence;
                DetectBox inBox(x1, y1, x2, y2, conf, cls);
                inBoxes.push_back(inBox);
            }
            DS->sort(frame, inBoxes);
            showDetection(frame, inBoxes, colors, classNames, videoWriter);
            frameIndex++;
        }
        videoWriter.release();
        cap.release();
        cv::destroyAllWindows();
    }

    void runInf(std::string videoPath) {
        std::cout << "runInf Called " << videoPath << "\n";
        cv::VideoCapture cap(videoPath); // Open the video file
        cv::VideoWriter videoWriter("NotusingDS.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, cv::Size(width * scale, height * scale));

        if (!cap.isOpened()) {
            std::cerr << "Error opening video file" << std::endl;
            return;
        } else {
            std::cout << "Video file opened successfully" << std::endl;
        }

        cv::Mat frame;
        int frameIndex = 1;
        while (cap.read(frame)) {
            if (interrupted) {
                break;
            }
            std::vector<Detection> output = yolo->runYolo(frame);

            int detections = output.size();
            std::cout << "Number of detections:" << detections << std::endl;

            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];

                cv::Rect box = detection.box;
                cv::Scalar color = detection.color;

                // Detection box
                cv::rectangle(frame, box, color, 2);

                // Detection box text
                std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(frame, textBox, color, cv::FILLED);
                cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            }
            // Inference ends here...

            // This is only for preview purposes
            cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
            cv::imshow("Inference", frame);
            videoWriter.write(frame);

            if (cv::waitKey(1) == 'q'){
                break;
            }
        }
        cap.release();
        videoWriter.release();
        cv::destroyAllWindows();
    }

    void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes, std::vector<cv::Scalar>& colors, std::vector<std::string>& classNames, cv::VideoWriter& videoWriter) {
        cv::Mat frame = img.clone();
        for (int i = 0; i < boxes.size(); i++) {
            // cout << "box: " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2 << " " << box.confidence << " " << box.classID << "\n";
            DetectBox box = boxes[i];
            cv::Point lt(box.x1, box.y1);
            cv::Point br(box.x2, box.y2);
            if (lt.x >= 0 && lt.y >= 0 && br.x <= img.cols && br.y <= img.rows) {
                cv::rectangle(frame, lt, br, colors[i], 2);
            }

            std::string lbl = classNames[i] + ' ' + "c: " + std::to_string(box.confidence).substr(0, 4) + ' ' + "t: " + std::to_string(box.trackID).substr(0, 4);
            cv::Size textSize = cv::getTextSize(lbl, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(lt.x, lt.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(frame, textBox, colors[i], cv::FILLED);
            cv::putText(frame, lbl, lt, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);
        videoWriter.write(frame);


        if (cv::waitKey(1) == 'q') {
            interrupted = true;
        }

    }

private:
    int convertHW(int num) {
        int Q = num / 32;
        int R = num % 32;
        std::cout << "Q: " << Q << " R: " << R << "\n";
        if (R == 0)
        {
            return Q * 32;
        } else
        {
            return 32 - (num - (32 * Q)) + num;
        }
    }

    int width;
    int height;
    float scale = 0.8;
    vector<DetectBox> out;
    DeepSort* DS;
    Yolo* yolo;
};

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    if (argc < 7) {
        std::cout << "./demo [input model path] [input video path] [yolo model path] [DS or Not DS] [model width] [model height]" << std::endl;
        return -1;
    }
    Tester* test = new Tester(argv[1], argv[3], argv[5], argv[6]);
    if (strcmp(argv[4], "DS") == 0) {
        test->run(argv[2]);
    } else      
    {
        test->runInf(argv[2]);
    }
    delete test;
    return 0;
}
