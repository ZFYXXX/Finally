#include <opencv2/opencv.hpp>  
#include <opencv2/objdetect.hpp>  
#include <iostream>  

int main(int argc, char** argv) {  
    // 检查命令行参数  
    if(argc != 2){  
        std::cout << "Usage: ./QRCodeDetectVideo <video_path>" << std::endl;  
        return -1;  
    }  

    // 打开视频文件或摄像头  
    cv::VideoCapture cap(argv[1]);  
    if(!cap.isOpened()){  
        std::cout << "无法打开视频源!" << std::endl;  
        return -1;  
    }  

    // 初始化QRCodeDetector  
    cv::QRCodeDetector qrDecoder;  

    cv::Mat frame;  
    while(cap.read(frame)){  
        if(frame.empty()){  
            std::cout << "无法读取帧!" << std::endl;  
            break;  
        }  

        // 检测并解码二维码  
        std::string data = qrDecoder.detectAndDecode(frame);  
        if(data.length() > 0){  
            std::cout << "解码数据: " << data << std::endl;  

            // 在图像中绘制二维码的位置（如果需要）  
            std::vector<cv::Point> bbox;  
            bool found = qrDecoder.detect(frame, bbox);  
            if(found && bbox.size() == 4){  
                for(int i = 0; i < 4; i++)  
                    cv::line(frame, bbox[i], bbox[(i+1)%4], cv::Scalar(0, 255, 0), 2);  
            }  

            // 在图像上显示解码内容  
            cv::putText(frame, data, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX,   
                        1.0, cv::Scalar(255, 0, 0), 2);  
        }  

        // 显示当前帧  
        cv::imshow("QR Code Detection in Video", frame);  

        // 按 'q' 键退出  
        if(cv::waitKey(1) == 'q') break;  
    }  

    cap.release();  
    cv::destroyAllWindows();  
    return 0;  
}