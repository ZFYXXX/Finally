#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

// RGB to NV12
void rgb_to_nv12(const vector<unsigned char>& rgb, int width, int height, vector<unsigned char>& nv12) {
    // 预留空间
    nv12.resize(width * height * 3 / 2);

    // Y平面
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = (i * width + j) * 3;
            unsigned char r = rgb[idx];
            unsigned char g = rgb[idx + 1];
            unsigned char b = rgb[idx + 2];
            nv12[i * width + j] = static_cast<unsigned char>((0.257 * r) + (0.504 * g) + (0.098 * b) + 16);
        }
    }

    // UV平面
    for (int i = 0; i < height / 2; ++i) {
        for (int j = 0; j < width / 2; ++j) {
            int idx = (i * 2 * width + j * 2) * 3;
            unsigned char r = rgb[idx];
            unsigned char g = rgb[idx + 1];
            unsigned char b = rgb[idx + 2];
            int uv_idx = width * height + i * width + j * 2;
            nv12[uv_idx] = static_cast<unsigned char>(-0.148 * r - 0.291 * g + 0.439 * b + 128); // U
            nv12[uv_idx + 1] = static_cast<unsigned char>(0.439 * r - 0.368 * g - 0.071 * b + 128); // V
        }
    }
}

// NV12 to JPG
void nv12_to_jpg(const vector<unsigned char>& nv12, int width, int height, const string& output_file) {
    // 创建一个NV12格式的Mat
    Mat nv12_mat(height * 3 / 2, width, CV_8UC1, const_cast<unsigned char*>(nv12.data()));

    // 将NV12转换为RGB
    Mat rgb_mat;
    cvtColor(nv12_mat, rgb_mat, COLOR_YUV2RGB_NV12);

    // 保存为JPG
    imwrite(output_file, rgb_mat);
}

int main() {
    string input_file = "pic1.jpg";
    string nv12_file = "pic1_nv12.yuv";
    string output_file = "pic1_reconstructed.jpg";

    // 读取JPG图像
    Mat image = imread(input_file, IMREAD_UNCHANGED);
    if (image.empty()) {
        cerr << "无法打开图像" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    vector<unsigned char> rgb_data(image.data, image.data + image.total() * image.elemSize());
    
    vector<unsigned char> nv12_data;
    
    // 开始计时
    auto start = chrono::high_resolution_clock::now();

    // RGB转NV12
    rgb_to_nv12(rgb_data, width, height, nv12_data);

    // 保存NV12文件
    ofstream nv12_out(nv12_file, ios::binary);
    nv12_out.write((char*)nv12_data.data(), nv12_data.size());
    nv12_out.close();

    // NV12转JPG
    nv12_to_jpg(nv12_data, width, height, output_file);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "处理时间: " << elapsed.count() << " ms" << endl;
    cout << "帧率: " << 1000.0 / elapsed.count() << " FPS" << endl;

    return 0;
}
