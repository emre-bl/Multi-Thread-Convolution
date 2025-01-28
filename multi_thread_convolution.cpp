#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>

constexpr int KERNEL_SIZE = 5;
constexpr int HISTOGRAM_SIZE = 256;
constexpr float KERNEL_NORMALIZER = 273.0f;

// Thread-safe histogram class - uses atomic integers
// Initializing bins to zero.
class Histogram
{
private:
    std::vector<int> bins;
    std::mutex histMutex; // only one thread can access the bins at a time

public:
    Histogram() : bins(HISTOGRAM_SIZE, 0) {} // Initialize bins to zero

    // Thread-safe increment method
    void increment(int index)
    {
        if (index >= 0 && index < HISTOGRAM_SIZE)
        {
            // Lock the mutex to ensure only one thread can access the bins at a time
            std::lock_guard<std::mutex> lock(histMutex);
            ++bins[index];
        }
    }

    // Returns a copy of the histogram data
    std::vector<int> getBins() const
    {
        return bins;
    }
};

// Kernel
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};

// Function to perform convolution for a specific region
void performConvolution(
    const cv::Mat &image,
    cv::Mat &output,
    Histogram &histogram,
    int startRow,
    int endRow)
{
    const int halfKernel = KERNEL_SIZE / 2;

    for (int i = startRow; i < endRow; ++i) // Rows
    {
        for (int j = halfKernel; j < image.cols - halfKernel; ++j) // Columns
        {
            float sum = 0.0f;

            for (int k = -halfKernel; k <= halfKernel; ++k) // Kernel rows
            {
                for (int l = -halfKernel; l <= halfKernel; ++l)
                {
                    sum += (float)image.at<uchar>(i + k, j + l) *
                           kernel[k + halfKernel][l + halfKernel];
                }
            }

            int result = static_cast<int>(sum / KERNEL_NORMALIZER);
            result = std::max(0, std::min(255, result)); // Safe clamping - [0, 255]

            output.at<uchar>(i, j) = static_cast<uchar>(result);
            histogram.increment(result); // Increment the histogram safely
        }
    }
}

// Function to save histogram as an image
void saveHistogramImage(const std::vector<int> &histData, const std::string &filename)
{
    const int scaleFactor = 4; // for better visualization
    const int histWidth = 256 * scaleFactor;
    const int histHeight = 300;

    // Find maximum value for scaling
    int maxVal = *std::max_element(histData.begin(), histData.end());

    // Create histogram image
    cv::Mat histImage(histHeight, histWidth, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < 256; ++i)
    {
        int height = static_cast<int>((histData[i] * (histHeight - 10.0)) / maxVal); // Çubuğun yüksekliği

        // extend the histogram vertically for better visualization
        cv::rectangle(
            histImage,
            cv::Point(i * scaleFactor, histHeight - 1),                    // Alt sol köşe
            cv::Point((i + 1) * scaleFactor - 1, histHeight - 1 - height), // Üst sağ köşe
            cv::Scalar(0),                                                 // Siyah renk
            cv::FILLED);                                                   // Dolu çubuklar
    }

    cv::imwrite(filename, histImage);
}

int main()
{
    // Image loading - grayscale
    cv::Mat image = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) // make sure image is loaded
    {
        std::cerr << "Error: Could not load image soru_3_gorsel.png" << std::endl;
        return -1;
    }

    // Create output image
    cv::Mat output = cv::Mat::zeros(image.size(), CV_8UC1);

    // Create histogram
    Histogram histogram;

    // Create threads
    std::vector<std::thread> threads;
    const int numThreads = 5;
    const int rowsPerThread = (image.rows - KERNEL_SIZE + 1) / numThreads; //

    // Launch threads
    for (int i = 0; i < numThreads; ++i)
    {
        int startRow = i * rowsPerThread + KERNEL_SIZE / 2;
        int endRow = (i == numThreads - 1) ? image.rows - KERNEL_SIZE / 2 : (i + 1) * rowsPerThread + KERNEL_SIZE / 2;

        threads.emplace_back(
            performConvolution,
            std::ref(image),
            std::ref(output),
            std::ref(histogram),
            startRow,
            endRow);
    }

    // Wait for all threads to complete
    for (auto &thread : threads)
    {
        thread.join();
    }

    // Save output image
    cv::imwrite("output_image.png", output);

    // Save histogram
    auto histogramData = histogram.getBins();
    saveHistogramImage(histogramData, "histogram.png");

    return 0;
}