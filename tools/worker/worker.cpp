#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "base64.h"
#include "commands.h"
#include "inbox.h"
#include "json.h"

// speech worker: serves one model over the worker protocol 2, JSON Lines on stdin and stdout, for another program
// that starts it, ASIST among them. Its stdout carries the protocol and nothing else. It loads the model, warmed up
// unless --no-warmup, adds the voices of --add-voice and says `ready` with the model's information, or `fatal` when it
// cannot; then it runs the requests one at a time in the order they become complete, but for info and count_tokens,
// which it answers as they arrive, each answered by exactly one terminal message, `end`, `error` or `cancelled`, and
// exits with 0 once stdin closes and every request is answered.

namespace {

using Clock = std::chrono::steady_clock;

/** The protocol of the running worker, which a failure that ends it reports `fatal` through. */
Protocol * current = nullptr;

/** The failure an exception is: the library's, or a defect, which the worker reports as internal. */
Failure failure_of(const std::exception & e) {
    if (const auto * f = dynamic_cast<const Failure *>(&e)) return *f;
    return Failure(speech_status_name(SPEECH_ERROR_INTERNAL), "", e.what());
}

std::string error_line(const std::string & id, const std::exception & e) {
    const Failure f = failure_of(e);
    return "{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + error_object(f.code(), f.option(), f.what()) + "}";
}

std::string cancelled_line(const std::string & id) {
    return "{\"type\":\"cancelled\",\"id\":" + json_string(id) + "}";
}

/**
 * What the callbacks of a running request need: the id, the next chunk's number and the time of the request's last
 * message, which progress waits a second after.
 */
struct Running {
    Protocol & protocol;
    const std::string & id;
    Clock::time_point last;
    int seq = 0;
    std::vector<int16_t> pcm;
};

int on_audio(const float * s, size_t n, void * user_data) {
    Running & r = *static_cast<Running *>(user_data);
    r.pcm.resize(n);
    for (size_t i = 0; i < n; i++) r.pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    r.protocol.line("{\"type\":\"chunk\",\"id\":" + json_string(r.id) + ",\"seq\":" + std::to_string(r.seq++) + ",\"pcm\":\"" +
                    base64((const uint8_t *) r.pcm.data(), n * sizeof(int16_t)) + "\"}");
    r.last = Clock::now();
    return 0;
}

int on_progress(double done, void * user_data) {
    Running & r = *static_cast<Running *>(user_data);
    const Clock::time_point now = Clock::now();
    if (now - r.last >= std::chrono::seconds(1)) {
        r.protocol.line("{\"type\":\"progress\",\"id\":" + json_string(r.id) + ",\"done\":" + json_number(done) + "}");
        r.last = now;
    }
    return 0;
}

/** The audio of transcribe or peek as the library takes it: 16-bit samples scaled as a 16-bit WAVE file is read. */
std::vector<float> samples_of(const Job & job) {
    std::vector<float> out(job.pcm.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = (float) job.pcm[i] / 32768.0f;
    return out;
}

void synthesize(Inbox & inbox, Protocol & protocol, speech_model * model, const Job & job) {
    const Request request = new_request(model);
    speech_request * r = request.get();
    if (!inbox.start(job, r)) return;
    Running running{protocol, job.id, Clock::now()};
    std::string terminal;
    try {
        check(speech_request_set_text(r, job.text.c_str()));
        apply_options(r, job.options);
        check(speech_request_set_progress(r, on_progress, &running));
        if (check(speech_synthesize(r, on_audio, &running)) == SPEECH_CANCELLED) {
            terminal = cancelled_line(job.id);
        } else {
            const speech_result * result = speech_request_result(r);
            terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"seed\":" + std::to_string(speech_result_seed(result)) +
                       ",\"samples\":" + std::to_string(speech_result_samples(result)) + ",\"stop\":\"" +
                       speech_stop_name(speech_result_stop(result)) + "\"}";
        }
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

/**
 * Recognizes the audio of a transcribe, answered by its terminal message, or of a peek, answered by a `partial` that
 * sends no progress and that a cancel of its request drops.
 */
void recognize(Inbox & inbox, Protocol & protocol, speech_model * model, const speech_model_info * info, const Job & job) {
    const bool peek = job.kind == Job::Kind::Peek;
    const Request request = new_request(model);
    speech_request * r = request.get();
    if (!inbox.start(job, r)) return;
    Running running{protocol, job.id, Clock::now()};
    std::string answer;
    try {
        const std::vector<float> samples = samples_of(job);
        check(speech_request_set_audio(r, samples.data(), samples.size(), job.sample_rate));
        apply_options(r, job.options);
        if (!peek) check(speech_request_set_progress(r, on_progress, &running));
        if (check(speech_transcribe(r)) == SPEECH_CANCELLED) {
            answer = peek ? "" : cancelled_line(job.id);
        } else {
            answer = std::string("{\"type\":\"") + (peek ? "partial" : "end") + "\",\"id\":" + json_string(job.id) +
                     recognition_members(speech_request_result(r), timestamps_in_effect(info, job.options)) + "}";
        }
    } catch (const std::exception & e) {
        const Failure f = failure_of(e);
        answer = peek ? "{\"type\":\"partial\",\"id\":" + json_string(job.id) + ",\"error\":" + error_object(f.code(), f.option(), f.what()) + "}"
                      : error_line(job.id, e);
    }
    if (peek) inbox.finish_peek(job, answer);
    else inbox.finish(job, answer);
}

/** Runs add_voice, which a cancel no longer stops once it runs. */
void add_voice(Inbox & inbox, speech_model * model, const Job & job) {
    if (!inbox.start(job, nullptr)) return;
    std::string terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + "}";
    try {
        check(speech_voice_add(model, job.name.c_str(), job.path.c_str()));
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

/**
 * The terminal message of an info, the model's information with the voices added so far, or of a count_tokens. Both
 * read information, which any thread may read while a request runs, so the reader answers them as they arrive.
 */
std::string answer_at_once(const speech_model * model, const speech_model_info * info, const Job & job) {
    try {
        if (job.kind == Job::Kind::Info) {
            return "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"model\":" + speech_model_info_json(model_info(model).get()) + "}";
        }
        size_t tokens = 0;
        check(speech_model_info_text_tokens(info, job.text.c_str(), &tokens));
        return "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"tokens\":" + std::to_string(tokens) + "}";
    } catch (const std::exception & e) {
        return error_line(job.id, e);
    }
}

int run_worker(const CommandLine & line, FILE * out) {
    // The reader thread reads stdin until it closes, which can outlive this function when a failure ends the worker,
    // so the protocol and the inbox it uses are never freed.
    Protocol & protocol = *new Protocol(out);
    current = &protocol;
    const Model model = load_model(line.args[0], line.loading(true));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    protocol.line("{\"type\":\"ready\",\"protocol\":2,\"version\":" + json_string(speech_version()) + ",\"model\":" + speech_model_info_json(m) + "}");
    std::fprintf(stderr, "speech worker: %s on %s, ready\n", speech_model_info_name(m), speech_model_info_device(m));

    const speech_model * loaded = model.get();
    Inbox & inbox = *new Inbox(protocol, speech_model_info_task(m), speech_model_info_name(m),
                               [loaded, m](const Job & job) { return answer_at_once(loaded, m, job); });
    std::thread reader([&inbox] { inbox.read(std::cin); });
    reader.detach();
    for (Job job; inbox.take(job);) {
        switch (job.kind) {
            case Job::Kind::Synthesize: synthesize(inbox, protocol, model.get(), job); break;
            case Job::Kind::Transcribe:
            case Job::Kind::Peek: recognize(inbox, protocol, model.get(), m, job); break;
            default: add_voice(inbox, model.get(), job); break;
        }
    }
    inbox.close_collecting();
    return 0;
}

void report_fatal(const Failure & e, FILE * out) {
    const std::string line = "{\"type\":\"fatal\",\"error\":" + error_object(e.code(), e.option(), e.what()) + "}";
    if (current) {
        current->line(line);
    } else {
        Protocol(out).line(line);
    }
}

}  // namespace

Command worker_command() {
    Command c;
    c.name = "worker";
    c.usage = "worker MODEL [options]";
    c.summary = "serve a model over JSON Lines on stdin and stdout (protocol 2)";
    c.description =
        "Loads MODEL, warmed up unless --no-warmup, adds the voices of --add-voice, and serves it over the worker protocol\n"
        "2: one JSON object per line on stdin and on stdout, and nothing else on stdout. The README gives the messages.";
    c.flags = {
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.min_args = 1;
    c.max_args = 1;
    c.run = run_worker;
    c.report = report_fatal;
    return c;
}
