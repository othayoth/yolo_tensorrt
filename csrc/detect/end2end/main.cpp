//
// Created by ubuntu on 1/20/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include "tqdm/tqdm.h"

const std::vector<std::string> CLASS_NAMES = {
   "mouse", "ball"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25}};

int main(int argc, char** argv)
{
    // cuda:0
    cudaSetDevice(0);

    // std::string buildInfo = cv::getBuildInformation();
    // std::cout << buildInfo << std::endl;

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 3);

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (IsFile(path)) 
    {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }

    cv::Mat             res, image;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::resizeWindow("result", 800, 2200/4);

    if (isVideo) {
        cv::VideoCapture cap(path);

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        printf("Total number of frames: %d\n", total_frames);
        
        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }

        std::string csv_path = path.substr(0, path.find_last_of('.')) + "_trt_track.csv";
        std::ofstream outputFile(csv_path, std::ios::app);
        long int frame_num = 0;
        // while (cap.read(image)) {
        for (int i : tqdm::range(total_frames)) 
        {   
            cap.read(image);
            frame_num++;
            
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            // yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf(" -- fps_infer %2.4lf Hz \t t_infer %2.4lf ms\n", 1000.0/tc, tc);

            
            if (outputFile.is_open()) {
                for (const auto& obj : objs) {

                    outputFile << frame_num << ","
                               << obj.label << "," << obj.prob << "," 
                               << obj.rect.x << "," << obj.rect.y << "," 
                               << obj.rect.width << "," << obj.rect.height << "\n";
                }
                
            } else {
                std::cerr << "Unable to open output file";
            }

            // cv::imshow("result", res);
            // if (cv::waitKey(10) == 'q') {
            //     break;
            // }
        }

        outputFile.close();
    }
    else {
        for (auto& path : imagePathList) {
            objs.clear();
            image = cv::imread(path);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
