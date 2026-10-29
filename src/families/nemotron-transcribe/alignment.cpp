#include "alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <tuple>

// The arithmetic here is NumPy's and PyTorch's, each product and sum rounded on its own: the build compiles this file
// with floating-point contraction off, so that no compiler fuses one into a multiply-add.

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";
constexpr double kInf = std::numeric_limits<double>::infinity();

/**
 * The positions torch.linspace(0, n - 1, steps) gives in float32 on the CPU, as its loop computes them: the step
 * (n - 1) / (steps - 1), the first half from the start and the rest back from the end, each in one rounding, as the
 * compiler of PyTorch's x86 kernels fuses end - step * k into a multiply-add. On the dumps every position agrees with
 * this, and a tenth of those of the second half do not without the fused rounding (2026-10-29).
 */
std::vector<float> linspace(float last, int64_t steps) {
    std::vector<float> out((size_t) steps);
    const float step = steps > 1 ? last / (float) (steps - 1) : 0.0f;
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; i++) out[(size_t) i] = i < halfway ? step * (float) i : std::fma(-step, (float) (steps - i - 1), last);
    return out;
}

/**
 * scipy.optimize.linear_sum_assignment() of the cost matrix [rows, cols] row-major: the column of each row, -1 for a
 * row left out, by the shortest augmenting paths of Crouse's algorithm as SciPy runs it, which breaks ties as SciPy does.
 */
std::vector<int> linear_sum_assignment(std::vector<double> cost, int rows, int cols) {
    std::vector<int> out((size_t) rows, -1);
    if (rows == 0 || cols == 0) return out;
    const bool transpose = cols < rows;
    int nr = rows, nc = cols;
    if (transpose) {
        std::vector<double> t(cost.size());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) t[(size_t) (j * rows + i)] = cost[(size_t) (i * cols + j)];
        }
        cost = std::move(t);
        std::swap(nr, nc);
    }
    std::vector<double> u((size_t) nr, 0), v((size_t) nc, 0), shortest((size_t) nc);
    std::vector<int> path((size_t) nc, -1), col4row((size_t) nr, -1), row4col((size_t) nc, -1), remaining((size_t) nc);
    std::vector<bool> sr((size_t) nr), sc((size_t) nc);
    for (int current = 0; current < nr; current++) {
        double min_value = 0;
        int i = current, sink = -1, left = nc;
        // The remaining columns in reverse, so that a constant matrix gives the identity, as SciPy fills them.
        for (int k = 0; k < nc; k++) remaining[(size_t) k] = nc - k - 1;
        std::fill(sr.begin(), sr.end(), false);
        std::fill(sc.begin(), sc.end(), false);
        std::fill(shortest.begin(), shortest.end(), kInf);
        while (sink == -1) {
            int index = -1;
            double lowest = kInf;
            sr[(size_t) i] = true;
            for (int k = 0; k < left; k++) {
                const int j = remaining[(size_t) k];
                const double r = min_value + cost[(size_t) (i * nc + j)] - u[(size_t) i] - v[(size_t) j];
                if (r < shortest[(size_t) j]) {
                    path[(size_t) j] = i;
                    shortest[(size_t) j] = r;
                }
                if (shortest[(size_t) j] < lowest || (shortest[(size_t) j] == lowest && row4col[(size_t) j] == -1)) {
                    lowest = shortest[(size_t) j];
                    index = k;
                }
            }
            min_value = lowest;
            if (min_value == kInf) return out;
            const int j = remaining[(size_t) index];
            if (row4col[(size_t) j] == -1) sink = j;
            else i = row4col[(size_t) j];
            sc[(size_t) j] = true;
            remaining[(size_t) index] = remaining[(size_t) --left];
        }
        u[(size_t) current] += min_value;
        for (int r = 0; r < nr; r++) {
            if (sr[(size_t) r] && r != current) u[(size_t) r] += min_value - shortest[(size_t) col4row[(size_t) r]];
        }
        for (int c = 0; c < nc; c++) {
            if (sc[(size_t) c]) v[(size_t) c] -= min_value - shortest[(size_t) c];
        }
        for (int j = sink;;) {
            const int r = path[(size_t) j];
            row4col[(size_t) j] = r;
            std::swap(col4row[(size_t) r], j);
            if (r == current) break;
        }
    }
    for (int r = 0; r < nr; r++) {
        if (transpose) out[(size_t) col4row[(size_t) r]] = r;
        else out[(size_t) r] = col4row[(size_t) r];
    }
    return out;
}

/** _ctc_anchor_band_starts(): the first state of the band at each searched frame. */
std::vector<int64_t> band_starts(const std::vector<int32_t> & greedy, const std::vector<int32_t> & target, int64_t states, int64_t half, int64_t width) {
    const int64_t frames = (int64_t) greedy.size();
    std::vector<int32_t> observed;
    std::vector<int64_t> times;
    for (int64_t t = 0; t < frames; t++) {
        if (greedy[(size_t) t] != 0 && (t == 0 || greedy[(size_t) t] != greedy[(size_t) t - 1])) {
            observed.push_back(greedy[(size_t) t]);
            times.push_back(t);
        }
    }
    auto trigrams = [](const std::vector<int32_t> & tokens) {
        std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<size_t>> table;
        for (size_t i = 0; i + 2 < tokens.size(); i++) table[{tokens[i], tokens[i + 1], tokens[i + 2]}].push_back(i);
        return table;
    };
    const auto left = trigrams(observed), right = trigrams(target);
    // Trigrams that occur once among the onsets and once in the target, as (onset of its middle token, its state).
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto & [key, at] : left) {
        const auto it = right.find(key);
        if (at.size() == 1 && it != right.end() && it->second.size() == 1) pairs.push_back({times[at[0] + 1], 2 * ((int64_t) it->second[0] + 1) + 1});
    }
    std::sort(pairs.begin(), pairs.end());
    // The longest chain increasing in state, by patience sorting with bisect_left.
    std::vector<int64_t> tails;
    std::vector<int> tail_ids, parent;
    for (size_t i = 0; i < pairs.size(); i++) {
        const size_t j = (size_t) (std::lower_bound(tails.begin(), tails.end(), pairs[i].second) - tails.begin());
        parent.push_back(j ? tail_ids[j - 1] : -1);
        if (j == tails.size()) {
            tails.push_back(pairs[i].second);
            tail_ids.push_back((int) i);
        } else {
            tails[j] = pairs[i].second;
            tail_ids[j] = (int) i;
        }
    }
    std::vector<std::pair<int64_t, int64_t>> chosen;
    for (int k = tail_ids.empty() ? -1 : tail_ids.back(); k >= 0; k = parent[(size_t) k]) chosen.push_back(pairs[(size_t) k]);
    std::reverse(chosen.begin(), chosen.end());
    std::vector<double> xs = {0}, ys = {0};
    for (const auto & [t, s] : chosen) {
        if (t > 0 && t < frames - 1) xs.push_back((double) t), ys.push_back((double) s);
    }
    xs.push_back((double) (frames - 1));
    ys.push_back((double) (states - 1));
    // np.interp() at each frame, then rint() and the clip, and the band moving at most two states a frame.
    std::vector<int64_t> starts((size_t) frames);
    for (int64_t t = 0; t < frames; t++) {
        const double x = (double) t;
        const size_t j = (size_t) (std::upper_bound(xs.begin(), xs.end(), x) - xs.begin()) - 1;
        double center;
        if (j == xs.size() - 1 || xs[j] == x) {
            center = ys[j];
        } else {
            const double slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
            const double run = x - xs[j];
            const double rise = slope * run;
            center = rise + ys[j];
        }
        starts[(size_t) t] = std::clamp((int64_t) std::nearbyint(center) - half, (int64_t) 0, states - width);
    }
    int64_t running = std::numeric_limits<int64_t>::max();
    for (int64_t t = 0; t < frames; t++) {
        running = std::min(running, starts[(size_t) t] - 2 * t);
        starts[(size_t) t] = std::max(running + 2 * t, std::max<int64_t>(0, states - width - 2 * (frames - 1 - t)));
    }
    return starts;
}

/** _banded_ctc_search() over `positions`, the path's state at each, all -1 with a score of -inf where none exists. */
StreamSearch search(const AlignmentInputs & in, int64_t columns, const std::vector<int32_t> & labels, std::vector<int64_t> positions,
                    const std::vector<float> & prior, const std::vector<int32_t> & greedy) {
    StreamSearch out;
    out.positions = std::move(positions);
    const int64_t frames = (int64_t) out.positions.size(), states = (int64_t) labels.size();
    const int64_t half = frames < 1000 ? states : std::min<int64_t>(states, (int64_t) std::ceil(0.10 * (double) states + 100));
    const int64_t width = std::min<int64_t>(states, 2 * half + 1);
    out.path.assign((size_t) frames, -1);
    out.score = -std::numeric_limits<float>::infinity();
    int64_t needed = 0;
    for (int64_t s = 1; s < states; s += 2) needed += 1 + (s >= 3 && labels[(size_t) s] == labels[(size_t) s - 2]);
    if (frames == 0 || frames < needed) return out;
    std::vector<int32_t> searched((size_t) frames), target;
    for (int64_t t = 0; t < frames; t++) searched[(size_t) t] = greedy[(size_t) out.positions[(size_t) t]];
    for (int64_t s = 1; s < states; s += 2) target.push_back(labels[(size_t) s]);
    out.starts = band_starts(searched, target, states, half, width);

    // The forward pass in float32, with stay (0), step (1) and skip (2) pointers, as _ctc_banded_forward_compiled().
    std::vector<uint8_t> can_skip((size_t) states, 0);
    for (int64_t s = 2; s < states; s++) can_skip[(size_t) s] = labels[(size_t) s] != 0 && labels[(size_t) s] != labels[(size_t) s - 2];
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> previous((size_t) width, ninf), current((size_t) width);
    std::vector<uint8_t> pointers((size_t) (frames * width), 0);
    const int64_t first = out.positions[0];
    for (int64_t j = 0; j < std::min<int64_t>(2, width); j++) {
        previous[(size_t) j] = in.log_probs[(size_t) (first * columns + labels[(size_t) j])];
        if (labels[(size_t) j] != 0) previous[(size_t) j] = previous[(size_t) j] + prior[(size_t) first];
    }
    for (int64_t t = 1; t < frames; t++) {
        const int64_t start = out.starts[(size_t) t], delta = start - out.starts[(size_t) t - 1], position = out.positions[(size_t) t];
        for (int64_t j = 0; j < width; j++) {
            const int64_t k = j + delta;
            float best = k < width ? previous[(size_t) k] : ninf;
            uint8_t move = 0;
            const float one = k - 1 >= 0 && k - 1 < width ? previous[(size_t) k - 1] : ninf;
            if (one > best) best = one, move = 1;
            const float two = k - 2 >= 0 && k - 2 < width ? previous[(size_t) k - 2] : ninf;
            if (two > best && can_skip[(size_t) (start + j)]) best = two, move = 2;
            pointers[(size_t) (t * width + j)] = move;
            const int32_t label = labels[(size_t) (start + j)];
            float value = best + in.log_probs[(size_t) (position * columns + label)];
            if (label != 0) value = value + prior[(size_t) position];
            current[(size_t) j] = value;
        }
        std::swap(previous, current);
    }
    // _ctc_banded_backtrace(): the better of the last two states that the last band holds, the last on a tie.
    int64_t end = -1;
    float score = ninf;
    for (const int64_t s : {states - 1, states - 2}) {
        const int64_t local = s - out.starts[(size_t) frames - 1];
        if (s < 0 || local < 0 || local >= width) continue;
        if (end < 0 || previous[(size_t) local] > score) end = s, score = previous[(size_t) local];
    }
    if (end < 0 || !std::isfinite(score)) return out;
    out.score = score;
    for (int64_t t = frames - 1, state = end; t >= 0; t--) {
        out.path[(size_t) t] = state;
        if (t) state -= pointers[(size_t) (t * width + state - out.starts[(size_t) t])];
    }
    return out;
}

}  // namespace

double round_ms(double x) {
    // x = m * 2^e exactly; x * 1000 rounded half to even, then divided, which IEEE division rounds once.
    if (!std::isfinite(x) || x == 0) return x;
    int e;
    const double f = std::frexp(std::fabs(x), &e);
    const uint64_t m = (uint64_t) std::ldexp(f, 53);
    const int shift = 53 - e;
    const uint64_t scaled = m * 1000;
    uint64_t q;
    if (shift <= 0) {
        q = scaled << -shift;
    } else if (shift >= 64) {
        q = 0;
    } else {
        q = scaled >> shift;
        const uint64_t rest = scaled & ((uint64_t(1) << shift) - 1), half = uint64_t(1) << (shift - 1);
        if (rest > half || (rest == half && (q & 1))) q++;
    }
    const double r = (double) q / 1000.0;
    return x < 0 ? -r : r;
}

Aligner::Aligner(const ModelFile & m)
    : speaker_threshold_(m.f32(kKeys + "align.speaker_activity_threshold")),
      speaker_weight_(m.f32(kKeys + "align.speaker_logprob_weight")),
      epsilon_(m.f32(kKeys + "align.epsilon")),
      maximum_token_len_(m.f32(kKeys + "align.maximum_token_len")),
      label_seconds_((double) m.u32(kKeys + "frontend.hop_length") / (double) m.u32("speech.sample_rate")),
      encoder_seconds_((double) (m.u32(kKeys + "frontend.hop_length") * m.u32(kKeys + "asr.subsampling_factor")) / (double) m.u32("speech.sample_rate")),
      max_word_frames_(m.size(kKeys + "align.max_word_frames") * m.size(kKeys + "ctc.upsampling_factor")) {}

Alignment Aligner::align(const AlignmentText & text, const AlignmentInputs & in) const {
    Alignment out;
    const int64_t frames = in.frames, columns = (int64_t) text.columns.size();
    const double step = in.duration / (double) frames;
    std::map<int32_t, int32_t> column_of;
    for (size_t c = 0; c < text.columns.size(); c++) column_of[text.columns[c]] = (int32_t) c;
    std::vector<std::vector<int32_t>> labels;
    for (const Stream & s : text.streams) {
        std::vector<int32_t> l;
        for (int32_t c : s.labels) l.push_back(column_of.at(c));
        int64_t needed = 0;
        for (size_t k = 1; k < l.size(); k += 2) needed += 1 + (k >= 3 && l[k] == l[k - 2]);
        if (needed > frames) {
            throw AlignmentFailure("the transcript of " + (s.speaker < 0 ? std::string("the words before the first speaker tag") : "speaker " + std::to_string(s.speaker)) +
                                   " holds more tokens than the audio's " + std::to_string(frames) + " CTC frames");
        }
        labels.push_back(std::move(l));
    }

    // The greedy column of each frame, the first of equal log probabilities as argmax() takes it.
    std::vector<int32_t> greedy((size_t) frames);
    for (int64_t t = 0; t < frames; t++) {
        const float * row = &in.log_probs[(size_t) (t * columns)];
        greedy[(size_t) t] = (int32_t) (std::max_element(row, row + columns) - row);
    }
    // _resample_speaker_probs() of the probabilities clamped to [0, 1].
    const int speakers = in.speakers;
    std::vector<float> probs(in.speaker_probs);
    for (float & p : probs) p = std::min(std::max(p, 0.0f), 1.0f);
    out.speaker.resize((size_t) (frames * speakers));
    if (in.speaker_frames == frames) {
        out.speaker = probs;
    } else if (in.speaker_frames == 1) {
        for (int64_t t = 0; t < frames; t++) std::copy(probs.begin(), probs.begin() + speakers, out.speaker.begin() + t * speakers);
    } else {
        const std::vector<float> at = linspace((float) (in.speaker_frames - 1), frames);
        for (int64_t t = 0; t < frames; t++) {
            const float p = at[(size_t) t], lower = std::floor(p), upper = std::ceil(p), fraction = p - lower, rest = 1.0f - fraction;
            for (int s = 0; s < speakers; s++) {
                const float a = probs[(size_t) ((int64_t) lower * speakers + s)] * rest;
                const float b = probs[(size_t) ((int64_t) upper * speakers + s)] * fraction;
                out.speaker[(size_t) (t * speakers + s)] = a + b;
            }
        }
    }

    // _one_pass_speaker_mapping(): the onsets of greedy tokens counted under each column, against each stream's tokens.
    const size_t streams = text.streams.size();
    out.columns.assign(streams, -1);
    if (streams > 0 && speakers > 0) {
        std::vector<double> observed((size_t) (speakers * columns), 0), expected(streams * (size_t) columns, 0);
        for (int64_t t = 0; t < frames; t++) {
            const int32_t g = greedy[(size_t) t];
            if (g == 0 || (t > 0 && g == greedy[(size_t) t - 1])) continue;
            for (int s = 0; s < speakers; s++) {
                if (out.speaker[(size_t) (t * speakers + s)] >= speaker_threshold_) observed[(size_t) (s * columns + g)] += 1;
            }
        }
        for (size_t s = 0; s < streams; s++) {
            for (size_t k = 1; k < labels[s].size(); k += 2) expected[s * (size_t) columns + (size_t) labels[s][k]] += 1;
        }
        std::vector<double> idf((size_t) columns);
        for (int64_t c = 0; c < columns; c++) {
            int seen = 0;
            for (int s = 0; s < speakers; s++) seen += observed[(size_t) (s * columns + c)] > 0;
            idf[(size_t) c] = std::log1p((double) speakers / (double) (1 + seen));
        }
        std::vector<double> a(expected.size()), b(observed.size());
        for (size_t i = 0; i < a.size(); i++) a[i] = std::sqrt(expected[i] * idf[i % (size_t) columns]);
        for (size_t i = 0; i < b.size(); i++) b[i] = std::sqrt(observed[i] * idf[i % (size_t) columns]);
        auto norm = [&](const std::vector<double> & m, size_t row) {
            double sum = 0;
            for (int64_t c = 0; c < columns; c++) sum += m[row * (size_t) columns + (size_t) c] * m[row * (size_t) columns + (size_t) c];
            return std::sqrt(sum);
        };
        std::vector<double> cost(streams * (size_t) speakers);
        for (size_t s = 0; s < streams; s++) {
            double e = 0;
            for (int64_t c = 0; c < columns; c++) e += expected[s * (size_t) columns + (size_t) c];
            for (int k = 0; k < speakers; k++) {
                double dot = 0, o = 0;
                for (int64_t c = 0; c < columns; c++) {
                    dot += a[s * (size_t) columns + (size_t) c] * b[(size_t) k * (size_t) columns + (size_t) c];
                    o += observed[(size_t) k * (size_t) columns + (size_t) c];
                }
                double similarity = dot / std::max(1e-8, norm(a, s) * norm(b, (size_t) k));
                similarity -= 0.1 * std::fabs(std::log((e + 1) / (o + 1)));
                cost[s * (size_t) speakers + (size_t) k] = -similarity;
            }
        }
        out.columns = linear_sum_assignment(cost, (int) streams, speakers);
    }

    // Each stream's search over the frames where its column's probability reaches 0.01.
    out.rows.assign(text.units.size(), {0, 0});
    for (size_t s = 0; s < streams; s++) {
        const int column = out.columns[s];
        std::vector<float> activity((size_t) frames, 0.0f);
        if (column >= 0) {
            for (int64_t t = 0; t < frames; t++) activity[(size_t) t] = out.speaker[(size_t) (t * speakers + column)];
        }
        std::vector<int64_t> positions;
        std::vector<float> prior((size_t) frames);
        for (int64_t t = 0; t < frames; t++) {
            if (activity[(size_t) t] >= 0.01f) positions.push_back(t);
            prior[(size_t) t] = speaker_weight_ * std::log(std::max(activity[(size_t) t], epsilon_));
        }
        out.searches.push_back(search(in, columns, labels[s], std::move(positions), prior, greedy));
        const StreamSearch & found = out.searches.back();
        const Stream & stream = text.streams[s];
        const std::string who = stream.speaker < 0 ? "the words before the first speaker tag" : "the words of speaker " + std::to_string(stream.speaker);
        if (!std::isfinite(found.score)) throw AlignmentFailure(who + " have no path through the frames where the speaker branch hears their speaker");
        // _word_rows_from_path(): a unit spans the frames whose state is one of its tokens.
        std::vector<int64_t> first(stream.units.size(), -1), last(stream.units.size(), -1);
        for (size_t t = 0; t < found.positions.size(); t++) {
            const int32_t k = stream.state_unit[(size_t) found.path[t]];
            if (k < 0) continue;
            if (first[(size_t) k] < 0) first[(size_t) k] = found.positions[t];
            last[(size_t) k] = found.positions[t];
        }
        for (size_t k = 0; k < stream.units.size(); k++) {
            if (first[k] < 0) throw AlignmentFailure("the CTC path skips the word " + text.units[stream.units[k]].text);
            const double start = (double) first[k] * step;
            out.rows[stream.units[k]] = {start, std::min((double) (last[k] + 1) * step, start + maximum_token_len_)};
        }
    }

    // diarization_segments(): each speaker's runs of 10 ms labels, ending at the audio's end at most.
    std::map<int, std::vector<std::pair<double, double>>> segments;
    for (int s = 0; s < in.label_speakers; s++) {
        for (int64_t t = 0; t < in.label_frames;) {
            if (!in.labels[(size_t) (s * in.label_frames + t)]) {
                t++;
                continue;
            }
            int64_t e = t;
            while (e < in.label_frames && in.labels[(size_t) (s * in.label_frames + e)]) e++;
            const double start = (double) t * label_seconds_, end = std::min((double) e * label_seconds_, in.duration);
            if (end > start) segments[s].push_back({start, end});
            t = e;
        }
    }
    for (auto & [s, list] : segments) std::sort(list.begin(), list.end());

    // _refine_word_timestamps(), stream by stream, its words by start.
    const double limit = (double) max_word_frames_ * std::min(step, encoder_seconds_);
    out.refined = out.rows;
    for (size_t s = 0; s < streams; s++) {
        std::vector<size_t> order = text.streams[s].units;
        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return out.rows[x].start < out.rows[y].start; });
        static const std::vector<std::pair<double, double>> none;
        const auto found = segments.find(out.columns[s]);
        const std::vector<std::pair<double, double>> & intervals = out.columns[s] >= 0 && found != segments.end() ? found->second : none;
        std::vector<int64_t> member(order.size(), -1);
        for (size_t i = 0; i < order.size(); i++) {
            const double start = out.rows[order[i]].start;
            const int64_t index = (int64_t) (std::upper_bound(intervals.begin(), intervals.end(), start,
                                                              [](double v, const std::pair<double, double> & r) { return v < r.first; }) -
                                             intervals.begin()) -
                                  1;
            member[i] = index >= 0 && start < intervals[(size_t) index].second ? index : -1;
        }
        std::vector<bool> seen(intervals.size(), false);
        for (size_t i = 0; i < order.size(); i++) {
            if (member[i] < 0 || seen[(size_t) member[i]]) continue;
            const double onset = intervals[(size_t) member[i]].first, gap = out.refined[order[i]].start - onset;
            if (gap > 0 && gap <= limit) out.refined[order[i]].start = onset;
            seen[(size_t) member[i]] = true;
        }
        for (size_t i = 0; i < order.size(); i++) {
            Span & w = out.refined[order[i]];
            double end;
            if (member[i] < 0) end = i + 1 < order.size() ? out.refined[order[i + 1]].start : w.end;
            else if (i + 1 < order.size() && member[i + 1] == member[i]) end = out.refined[order[i + 1]].start;
            else end = intervals[(size_t) member[i]].second;
            w.end = std::max(w.start, std::min(end, w.start + limit));
        }
    }
    for (const Span & w : out.refined) out.times.push_back({round_ms(w.start), round_ms(w.end)});
    return out;
}

}  // namespace nemotron_transcribe
