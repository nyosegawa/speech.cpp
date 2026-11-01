#include "audio-stream.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace server {

namespace {

/** The audio of a request as the worker sends it: 16-bit little-endian samples, clamped to [-1, 1] and rounded. */
void append_pcm(std::string & out, const float * s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
        out += (char) (v & 0xFF);
        out += (char) ((uint16_t) v >> 8);
    }
}

}  // namespace

int on_job_audio(const float * s, size_t n, void * user_data) {
    AudioJob & job = *static_cast<AudioJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    if (job.samples == 0) job.first_audio = Clock::now();
    append_pcm(job.pending, s, n);
    job.samples += n;
    job.changed.notify_all();
    return 0;
}

int on_job_progress(double, void * user_data) {
    AudioJob & job = *static_cast<AudioJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    job.changed.notify_all();
    return 0;
}

std::string wav_header(size_t data_bytes, int sample_rate) {
    std::string h;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) h += (char) ((v >> (8 * i)) & 0xFF); };
    auto u16 = [&](uint16_t v) { h += (char) (v & 0xFF); h += (char) (v >> 8); };
    h += "RIFF";
    u32((uint32_t) (36 + data_bytes));
    h += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(1);
    u32((uint32_t) sample_rate);
    u32((uint32_t) sample_rate * 2);
    u16(2);
    u16(16);
    h += "data";
    u32((uint32_t) data_bytes);
    return h;
}

bool stream_audio(AudioJob & job, httplib::DataSink & sink, const std::function<std::string(const std::string & pcm)> & frame,
                  const std::function<std::string()> & ending) {
    std::unique_lock<std::mutex> lock(job.mutex);
    for (;;) {
        if (!job.pending.empty()) {
            std::string pcm;
            pcm.swap(job.pending);
            lock.unlock();
            const std::string out = frame(pcm);
            // A write to a socket the client has closed succeeds once more before it fails, so a client that went
            // away is noticed a chunk earlier by looking first.
            if (!sink.is_writable() || !sink.write(out.data(), out.size())) {
                job.abandon();
                return false;
            }
            lock.lock();
            continue;
        }
        if (job.finished) break;
        if (job.changed.wait_for(lock, POLL) == std::cv_status::timeout && !job.finished && job.pending.empty()) {
            lock.unlock();
            if (!sink.is_writable()) {
                job.abandon();
                return false;
            }
            lock.lock();
        }
    }
    lock.unlock();
    const std::string last = ending();
    if (!last.empty() && !sink.write(last.data(), last.size())) return false;
    lock.lock();
    const bool ok = job.status == SPEECH_OK;
    lock.unlock();
    // A failed run's stream ends broken, unless its last message told the client of the failure.
    if (!ok && last.empty()) return false;
    sink.done();
    return true;
}

}  // namespace server
