#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "logging.h"
#include <ctime>

using std::vector;

static Logger gLogger;

void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cout << "box: " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2 << " " << box.confidence << " " << box.classID << "\n";
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        if (lt.x >= 0 && lt.y >= 0 && br.x <= img.cols && br.y <= img.rows) {
            cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        }
        std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,255,0));
    }
    cv::imshow("DeepSort", temp);
    if (cv::waitKey(1) == 'q') {
        return;
    }
}

class Tester {
public:
    Tester(std::string modelPath) {
        allDetections.clear();
        out.clear();
        DS = new DeepSort(modelPath, 128, 256, 0, &gLogger);
        std::cout << "DeepSort initialized!" << "\n";
    }
    ~Tester() {
        delete DS;
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

    void loadDetections(std::string txtPath) {
        this->txtPath = txtPath;
        std::cout << "Loading detections from " << txtPath << "\n";
        std::ifstream inFile;
        inFile.open(txtPath, std::ios::binary);
        std::string temp;
        vector<std::string> token;
        while (std::getline(inFile, temp)) {
            split(temp, token, ' ');
            int frame = std::atoi(token[0].c_str());
            int c     = std::atoi(token[1].c_str());
            int x     = std::atoi(token[2].c_str());
            int y     = std::atoi(token[3].c_str());
            int w     = std::atoi(token[4].c_str());
            int h     = std::atoi(token[5].c_str());
            float con = std::atof(token[6].c_str());     
            while (allDetections.size() <= frame) {
                vector<DetectBox> t;
                allDetections.push_back(t);
            }
            DetectBox dd(x-w/2, y-h/2, x+w/2, y+h/2, con, c);
            allDetections[frame].push_back(dd);
        }
        allDetections.pop_back();
        std::cout << "Loading detections complete! \n";
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
        while (cap.read(frame)) {
            if (frameIndex >= allDetections.size()) break;
            cv::Mat temp = frame.clone();
            cv::cvtColor(frame, temp, cv::COLOR_BGR2RGB);
            vector<DetectBox> d = allDetections[frameIndex];
            // DS->sort(temp, d);
            showDetection(frame, d);
            frameIndex++;
        }
        cap.release();
        cv::destroyAllWindows();
    }

private:
    vector<vector<DetectBox>> allDetections;
    vector<DetectBox> out;
    std::string txtPath;
    DeepSort* DS;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "./demo [input model path] [input txt path] [input video path]" << std::endl;
        return -1;
    }
    Tester* test = new Tester(argv[1]);
    test->loadDetections(argv[2]);
    test->run(argv[3]);
    delete test;
    return 0;
}
