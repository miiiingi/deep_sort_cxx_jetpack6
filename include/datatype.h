#ifndef DATATYPE_H
#define DATATYPE_H

/**
 * @brief DetectBox 생성자
 * 
 * @param x1 바운딩 박스의 왼쪽 상단 x 좌표 (기본값: 0)
 * @param y1 바운딩 박스의 왼쪽 상단 y 좌표 (기본값: 0)
 * @param x2 바운딩 박스의 오른쪽 하단 x 좌표 (기본값: 0)
 * @param y2 바운딩 박스의 오른쪽 하단 y 좌표 (기본값: 0)
 * @param confidence 탐지 신뢰도 (기본값: 0)
 * @param classID 클래스 ID (기본값: -1)
 * @param trackID 트랙 ID (기본값: -1)
 */
typedef struct DetectBox {
    DetectBox(float x1=0, float y1=0, float x2=0, float y2=0, 
            float confidence=0, float classID=-1, float trackID=-1) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->confidence = confidence;
        this->classID = classID;
        this->trackID = trackID;
    }
    float x1, y1, x2, y2;
    float confidence;
    float classID;
    float trackID;
} DetectBox;

#endif // DATATYPE_H

#ifndef DEEPSORTDATATYPE_H
#define DEEPSORTDATATYPE_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
typedef struct CLSCONF {
    CLSCONF() {
        this->cls = -1;
        this->conf = -1;
    }
    CLSCONF(int cls, float conf) {
        this->cls = cls;
        this->conf = conf;
    }
    int cls;
    float conf;
} CLSCONF;

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 256, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

#endif //DEEPSORTDATATYPE_H