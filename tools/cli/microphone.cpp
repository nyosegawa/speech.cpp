#include "microphone.h"

#include "failure.h"
#include "miniaudio.h"

struct Microphone::Device {
    ma_device device;
};

namespace {

Failure microphone_failure(const std::string & what, ma_result result) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot " + what + " the default microphone: " + ma_result_description(result) +
                                                                "; check that one is connected and turned on");
}

}  // namespace

Microphone::Microphone() : device_(new Device) {
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    // The device's own rate, which the library resamples as it does a file's, rather than miniaudio's resampler.
    config.sampleRate = 0;
    config.pUserData = this;
    config.dataCallback = [](ma_device * device, void *, const void * input, ma_uint32 frames) {
        static_cast<Microphone *>(device->pUserData)->captured(static_cast<const float *>(input), frames);
    };
    if (const ma_result r = ma_device_init(nullptr, &config, &device_->device); r != MA_SUCCESS) throw microphone_failure("open", r);
    rate_ = (int) device_->device.sampleRate;
    char name[MA_MAX_DEVICE_NAME_LENGTH + 1] = {0};
    ma_device_get_name(&device_->device, ma_device_type_capture, name, sizeof name, nullptr);
    name_ = name;
    if (const ma_result r = ma_device_start(&device_->device); r != MA_SUCCESS) {
        ma_device_uninit(&device_->device);
        throw microphone_failure("start", r);
    }
}

Microphone::~Microphone() {
    ma_device_uninit(&device_->device);
}

void Microphone::captured(const float * samples, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    captured_.insert(captured_.end(), samples, samples + n);
    arrived_.notify_all();
}

std::vector<float> Microphone::take(std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    arrived_.wait_for(lock, wait, [&] { return !captured_.empty(); });
    std::vector<float> out;
    out.swap(captured_);
    return out;
}
