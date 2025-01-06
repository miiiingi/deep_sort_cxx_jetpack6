#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "yolo.h"
#include "logging.h"
#include <ctime>
#include <csignal>

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
    Tester(std::string modelPath, std::string yoloPath) {
        out.clear();
        DS = new DeepSort(modelPath, 128, 256, 0, &gLogger);
        std::cout << "DeepSort initialized!" << "\n";
        yolo = new Yolo(yoloPath, cv::Size(1920, 1088), "classes.txt", true);
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
        std::cout << "Running DeepSort on " << videoPath << "\n";
        cv::VideoCapture cap(videoPath); // Open the video file
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
            std::cout << "result size: " << DS->result.size() << "\n";
            std::cout << "results: " << DS->results.size() << "\n";
            for (int i = 0; i < DS->result.size(); i++)
            {
                int x1 = DS->result[i].second(0,0);
                int y1 = DS->result[i].second(0,1);
                int x2 = DS->result[i].second(0,2);
                int y2 = DS->result[i].second(0,3);
                int trackId = DS->result[i].first;
                int cls = DS->results[i].first.cls;
                float conf = DS->results[i].first.conf;
                DetectBox outBox(x1, y1, x2, y2, conf, cls, trackId);
                outBoxes.push_back(outBox);
            }
            showDetection(frame, outBoxes, colors, classNames);
            // if (frameIndex >= 1)
            // {
            //     break;
            // }
            
            frameIndex++;
        }
        cap.release();
        cv::destroyAllWindows();
    }

    void runInf(std::string videoPath) {
        std::cout << "Running DeepSort on " << videoPath << "\n";
        cv::VideoCapture cap(videoPath); // Open the video file
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
            float scale = 0.8;
            cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
            cv::imshow("Inference", frame);

            if (cv::waitKey(1) == 'q'){
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }

    void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes, std::vector<cv::Scalar>& colors, std::vector<std::string>& classNames) {
        cv::Mat frame = img.clone();
        for (int i = 0; i < boxes.size(); i++) {
            // cout << "box: " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2 << " " << box.confidence << " " << box.classID << "\n";
            DetectBox box = boxes[i];
            cv::Point lt(box.x1, box.y1);
            cv::Point br(box.x2, box.y2);
            if (lt.x >= 0 && lt.y >= 0 && br.x <= img.cols && br.y <= img.rows) {
                cv::rectangle(frame, lt, br, colors[i], 2);
            }


            // std::string lbl = cv::format("ID:%d_C:%s_CONF:%.2f", (int)box.trackID, classNames[i], box.conf);
            std::string lbl = classNames[i] + ' ' + std::to_string(box.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(lbl, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(lt.x, lt.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(frame, textBox, colors[i], cv::FILLED);
            cv::putText(frame, lbl, lt, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);


        if (cv::waitKey(1) == 'q') {
            interrupted = true;
        }

    }

private:
    vector<DetectBox> out;
    std::string txtPath;
    DeepSort* DS;
    Yolo* yolo;
};

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    if (argc < 4) {
        std::cout << "./demo [input model path] [input video path] [yolo model path]" << std::endl;
        return -1;
    }
    Tester* test = new Tester(argv[1], argv[3]);
    test->run(argv[2]);
    // test->runInf(argv[3]);
    delete test;
    return 0;
}
