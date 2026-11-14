#include "speaker.h"

#include <algorithm>

#include "failure.h"
#include "miniaudio.h"

struct Speaker::Device {
    ma_device device;
};

namespace {

Failure speaker_failure(const std::string & what, ma_result result) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot " + what + " the default audio output: " + ma_result_description(result) +
                                                                "; check that one is connected and turned on");
}

}  // namespace

Speaker::Speaker(int rate) : device_(new Device) {
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    // miniaudio resamples to the device's rate: the library resamples only what a model hears, and this is heard by the
    // listener alone.
    config.sampleRate = (ma_uint32) rate;
    config.pUserData = this;
    config.dataCallback = [](ma_device * device, void * output, const void *, ma_uint32 frames) {
        static_cast<Speaker *>(device->pUserData)->fill(static_cast<float *>(output), frames);
    };
    if (const ma_result r = ma_device_init(nullptr, &config, &device_->device); r != MA_SUCCESS) throw speaker_failure("open", r);
    char name[MA_MAX_DEVICE_NAME_LENGTH + 1] = {0};
    ma_device_get_name(&device_->device, ma_device_type_playback, name, sizeof name, nullptr);
    name_ = name;
    if (const ma_result r = ma_device_start(&device_->device); r != MA_SUCCESS) {
        ma_device_uninit(&device_->device);
        throw speaker_failure("start", r);
    }
}

Speaker::~Speaker() {
    ma_device_uninit(&device_->device);
}

void Speaker::play(const float * samples, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.insert(queue_.end(), samples, samples + n);
}

size_t Speaker::queued() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void Speaker::fill(float * out, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t have = std::min(n, queue_.size());
    std::copy(queue_.begin(), queue_.begin() + (std::ptrdiff_t) have, out);
    queue_.erase(queue_.begin(), queue_.begin() + (std::ptrdiff_t) have);
    std::fill(out + have, out + n, 0.0f);
}
