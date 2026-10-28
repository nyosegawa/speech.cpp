// Checks the update of Sortformer's speaker cache and FIFO queue against NeMo's SortformerModules.streaming_update() on
// every step of each configuration of reference/sortformer/dump.py, each from the step's own input: the official state
// before the step, the chunk's official pre-encoded frames and the official 80 ms probabilities the step gave its input.
// The state after the step must be the official one exactly: the same frames, or the silence embedding, in the same rows
// of the cache and the queue, with the same probabilities, compressed alike; but where the compression chose among equal
// scores, torch.topk() leaves the choice to its C++ library's std::nth_element(), which libc++ and libstdc++ make
// differently, and a step that differs there is counted apart. The dumps are those in
// <reference out dir>/<the file's general.name>/.
//
// usage: sortformer-cache-check <model.gguf> <reference out dir>

#include <cstdio>
#include <string>

#include "args.h"
#include "backend.h"
#include "ggml-cpu.h"
#include "sortformer-dumps.h"
#include "sortformer/diarizer.h"

using namespace sortformer;
using namespace sortformer_dumps;

namespace {

/** The frames of `rows` named by the first frame of the same bytes, or -1 for the silence embedding. */
std::vector<int64_t> canonical(const std::vector<int64_t> & rows, const States & states) {
    std::vector<int64_t> out;
    for (const int64_t r : rows) out.push_back(r < 0 ? -1 : states.canonical.at((size_t) r));
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        // The update runs on the host; the backend only holds the model.
        ggml_backend_t backend = ggml_backend_cpu_init();
        bool ok = true;
        {
            Diarizer diarizer(args[1], backend);
            const int speakers = diarizer.speakers();
            for (const auto & input : reference_dumps(args[2], diarizer.model())) {
                for (const Config & c : configs(input)) {
                    const States states(c, speakers);
                    const int64_t d = states.d();
                    int64_t compressions = 0, wrong = 0, tied = 0;
                    std::string first_wrong;
                    for (size_t k = 0; k < c.steps.size(); k++) {
                        const Step & st = c.steps[k];
                        State state = states.after(c, (int64_t) k - 1, diarizer.silence(), speakers);
                        const State want = states.after(c, (int64_t) k, diarizer.silence(), speakers);
                        int ties = 0;
                        update(state, c.streaming, diarizer.compression(), diarizer.silence(), &states.table.f32[(size_t) (st.first_frame * d)],
                               st.chunk_frames, st.first_frame, &states.step_probs.f32[(size_t) (states.step_offset[k] * speakers)], 0, &ties);
                        // The cache was compressed where the rows the queue gave it took it past spkcache_len.
                        if (st.cache_before + st.fifo_before + st.chunk_frames - st.fifo_after > c.streaming.spkcache_len) compressions++;
                        const bool same = canonical(state.cache_frames, states) == want.cache_frames && canonical(state.fifo_frames, states) == want.fifo_frames &&
                                          state.cache_probs == want.cache_probs && state.fifo_probs == want.fifo_probs && state.compressed == want.compressed;
                        if (!same && ties > 0) {
                            tied++;
                        } else if (!same) {
                            if (wrong == 0) {
                                int64_t rows = 0;
                                const std::vector<int64_t> got = canonical(state.cache_frames, states);
                                for (size_t i = 0; i < std::min(got.size(), want.cache_frames.size()); i++) rows += got[i] != want.cache_frames[i];
                                first_wrong = "step " + std::to_string(k) + ": " + std::to_string(rows) + " of " + std::to_string(want.cache_frames.size()) +
                                              " rows of the cache name other frames, the cache has " + std::to_string(got.size()) + " rows";
                            }
                            wrong++;
                        }
                    }
                    std::printf("%s %s: %zu steps, %lld compressions, %lld steps that differ where scores tie, %lld steps whose state differs%s%s\n",
                                input.filename().u8string().c_str(), c.name.c_str(), c.steps.size(), (long long) compressions, (long long) tied, (long long) wrong,
                                wrong ? ", first at " : "", first_wrong.c_str());
                    ok = ok && wrong == 0;
                }
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
