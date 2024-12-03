// detect.cpp  
#include <iostream>  
#include <vector>  
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>  

int main() {  
    // 初始化ONNX Runtime环境  
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");  
    Ort::SessionOptions session_options;  
    session_options.SetIntraOpNumThreads(1);  

    // 启用GPU  
    #ifdef USE_CUDA  
    OrtCUDAProviderOptions cuda_options;  
    session_options.AppendExecutionProvider_CUDA(cuda_options);  
    #endif  

    // 创建会话  
    const char* model_path = "qrcode_model.onnx";  
    Ort::Session session(env, model_path, session_options);  

    // 获取输入输出信息  
    Ort::AllocatorWithDefaultOptions allocator;  
    auto input_name = session.GetInputName(0, allocator);  
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  
    auto output_name = session.GetOutputName(0, allocator);  

    // 读取并预处理图像  
    cv::Mat img = cv::imread("test_qrcode.jpg");  
    if (img.empty()) {  
        std::cerr << "Image not found!" << std::endl;  
        return -1;  
    }  
    cv::resize(img, img, cv::Size(32, 32));  
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  
    img.convertTo(img, CV_32F, 1.0 / 255);  

    // 转换为一维数组并添加批次维度  
    std::vector<float> input_tensor_values;  
    for(int y = 0; y < img.rows; y++) {  
        for(int x = 0; x < img.cols; x++) {  
            for(int c = 0; c < img.channels(); c++) {  
                input_tensor_values.push_back(img.at<cv::Vec3f>(y,x)[c]);  
            }  
        }  
    }  
    std::vector<int64_t> input_shape_vec = {1, 3, 32, 32};  
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);  
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape_vec.data(), input_shape_vec.size());  

    // 运行推理  
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);  

    // 获取结果  
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();  
    // 假设是分类任务，找到最大值索引  
    int max_idx = 0;  
    float max_val = floatarr[0];  
    for(int i = 1; i < 10; i++) { // num_classes=10  
        if(floatarr[i] > max_val){  
            max_val = floatarr[i];  
            max_idx = i;  
        }  
    }  
    std::cout << "Predicted class: " << max_idx << " with confidence " << max_val << std::endl;  

    return 0;  
}