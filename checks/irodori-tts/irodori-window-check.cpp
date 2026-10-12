// Checks the rule by which Irodori-TTS's decoder sizes its windows (next_window()): each way it decides, the first
// window of a speech shorter than it among them, and simulated machines that decode a frame in a set time: one as fast
// as an Apple M5, one twice as fast, one half as fast, one that slows down after its second window, one that slows down
// once its windows have grown, and one slower than real time. Each speaks a sentence of 9 frames, one of 145 (5.8 s) and
// one of 685 (27.4 s) as Codec::decode() would, its estimate of the time per frame made as decode() makes it, and the
// listener plays from the first window on, under the synthesizer's rule and under the same rule held to its ceiling. It
// fails unless the first window is the rule's or the frames left; every later one is from the floor to the largest or
// the frames left, and one past the ceiling was expected to decode within the rule's wait; on no machine does the
// listener ever hold less audio when a window arrives than the least it holds under the rule held to its ceiling, nor
// does the rule decode more frames than that one; the half as fast machine never runs dry where windows of the ceiling
// do; the ones that slow down never run dry; and the one slower than real time takes the ceiling.
//
// usage: irodori-window-check

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "irodori-tts/synthesizer.h"

using namespace irodori;

namespace {

/** A machine: the seconds it takes to decode a frame, a window's margins included, at a time from the start. */
struct Machine {
    const char * name;
    std::function<double(double now)> seconds_per_frame;
};

/**
 * The windows of one sentence, the least audio the listener had when a window arrived, the time it had none, the frames
 * decoded, margins included, and whether a window past the ceiling was expected to decode for longer than the wait.
 */
struct Played {
    std::vector<int> windows;
    double least = INFINITY, dry = 0;
    int64_t decoded = 0;
    bool over_wait = false;
};

/** One sentence under `rule`, or in windows of its ceiling after the first when `fixed`. */
Played play(const WindowRule & rule, const Machine & machine, int64_t frames, bool fixed) {
    Played out;
    double now = 0, first_sent = 0, sent = 0, estimate = 0;
    for (int64_t a = 0; a < frames;) {
        const DecodeProgress progress{a, frames - a, sent, a == 0 ? 0 : now - first_sent, estimate};
        const int window = fixed && a > 0 ? (int) std::min<int64_t>(rule.ceiling, frames - a) : next_window(rule, progress);
        out.windows.push_back(window);
        const int64_t from = std::max<int64_t>(0, a - rule.context), to = std::min(frames, a + window + rule.context);
        if (a > 0 && window > rule.ceiling) out.over_wait = out.over_wait || estimate * (double) (window + 2 * rule.context) > rule.wait();
        const double per_frame = machine.seconds_per_frame(now);
        now += per_frame * (double) (to - from);
        out.decoded += to - from;
        estimate = next_estimate(estimate, per_frame);
        if (a == 0) {
            first_sent = now;
        } else {
            const double left = sent - (now - first_sent);
            out.least = std::min(out.least, left);
            if (left < 0) out.dry -= left;
        }
        sent += window * rule.frame_seconds;
        a += window;
    }
    return out;
}

std::string list(const std::vector<int> & windows) {
    std::string out;
    for (int w : windows) out += (out.empty() ? "" : " ") + std::to_string(w);
    return out;
}

void print(const char * what, const Played & p) {
    std::printf("  %s %s; least in hand %.3f s, dry %.3f s, %lld frames decoded\n", what, list(p.windows).c_str(), p.least, p.dry,
                (long long) p.decoded);
}

}  // namespace

int main() {
    // The synthesizer's rule for Irodori-TTS's codec: frames of 1920 samples at 48 kHz, and a decoder whose receptive
    // field reaches 8 frames.
    const WindowRule rule = Synthesizer::window_rule(1920, 48000, 8);
    WindowRule held = rule;
    held.largest = held.ceiling;
    // An Apple M5 decoded the 64 frames of a 48-frame window in 0.256 s on Metal with v4.1-Small-MF's F16 file, and
    // windows from 12 to 256 frames at 4.0 ms a frame (2026-10-13).
    const double m5 = 0.004;
    const Machine machines[] = {
        {"as fast as an Apple M5", [&](double) { return m5; }},
        {"twice as fast", [&](double) { return m5 / 2; }},
        {"half as fast", [&](double) { return 2 * m5; }},
        // From its third window on, which starts after 0.3 s, 27 ms a frame: faster than real time with 48 frames and
        // slower with 24.
        {"as fast, then slower after its second window", [&](double now) { return now < 0.3 ? m5 : 0.027; }},
        // As slow from 2 s on, while a window past the ceiling decodes.
        {"as fast, then slower once its windows have grown", [&](double now) { return now < 2 ? m5 : 0.027; }},
        {"slower than real time", [&](double) { return 0.06; }},
    };
    bool ok = true;
    // Each way the rule decides, on a rule of its own: 12 frames first, then 24 to 48 frames and up to 96 within its
    // wait of 0.38 s and the audio in hand, frames of 0.04 s, margins of 10, 0.1 s in hand.
    const WindowRule own{12, 24, 48, 96, 10, 0.04, 0.1};
    const struct {
        const char * what;
        DecodeProgress progress;
        int want;
    } cases[] = {
        {"the first window", {0, 100, 0, 0, 0}, 12},
        {"the first window of a speech shorter than it", {0, 9, 0, 0, 0}, 9},
        {"audio in hand past the ceiling, as much as decodes within the wait", {12, 100, 4.4, 0.4, 0.005}, 56},
        {"audio in hand past the ceiling, as much as the listener holds", {12, 100, 3.4, 0.4, 0.001}, 52},
        {"audio in hand past the largest within the wait", {12, 100, 6.4, 0.4, 0.001}, 96},
        {"audio in hand, with the ceiling decoding for longer than the wait", {12, 100, 6.4, 0.4, 0.01}, 48},
        {"time for the ceiling within the wait, with too little audio in hand to pass it", {12, 100, 2.4, 0.4, 0.005}, 48},
        {"time for 33 frames and some", {12, 100, 0.48, 0.0, 0.0071}, 33},
        {"no time for the floor, decoding faster than real time at it", {12, 100, 0.48, 0.45, 0.005}, 24},
        {"no time for the floor, decoding slower than real time at it", {12, 100, 0.48, 0.45, 0.03}, 48},
        {"fewer frames left than the floor", {60, 10, 0.48, 0.0, 0.005}, 10},
        {"fewer frames left than fit", {60, 30, 2.4, 0.4, 0.005}, 30},
    };
    for (const auto & c : cases) {
        const int got = next_window(own, c.progress);
        std::printf("%s: %d frames (want %d)\n", c.what, got, c.want);
        ok = ok && got == c.want;
    }
    for (int64_t frames : {9, 145, 685}) {
        for (const Machine & m : machines) {
            const Played adaptive = play(rule, m, frames, false), capped = play(held, m, frames, false), fixed = play(rule, m, frames, true);
            std::printf("%lld frames, %s:\n", (long long) frames, m.name);
            print("by the rule", adaptive);
            print("held to the ceiling", capped);
            print("the ceiling a window", fixed);
            int64_t left = frames;
            for (size_t i = 0; i < adaptive.windows.size(); i++) {
                const int w = adaptive.windows[i];
                ok = ok && (i == 0 ? w == std::min<int64_t>(rule.first, frames) : (w >= rule.floor && w <= rule.largest) || w == left);
                left -= w;
            }
            ok = ok && !adaptive.over_wait && adaptive.least >= capped.least && adaptive.decoded <= capped.decoded;
            // A speech within the first window has no later ones to size.
            if (frames <= rule.first) continue;
            const std::string name = m.name;
            if (name == "half as fast") ok = ok && adaptive.dry == 0 && fixed.dry > 0;
            if (name.rfind("as fast, then slower", 0) == 0) ok = ok && adaptive.dry == 0;
            if (name == "slower than real time") {
                for (size_t i = 1; i + 1 < adaptive.windows.size(); i++) ok = ok && adaptive.windows[i] == rule.ceiling;
            }
        }
    }
    std::printf(ok ? "the rule holds on every machine\n" : "FAIL: the windows break the rule on a machine above\n");
    return ok ? 0 : 1;
}
