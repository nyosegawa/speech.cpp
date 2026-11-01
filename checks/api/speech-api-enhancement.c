/*
 * The enhancement part of speech-api-check: the C API with an enhancement model, through the shared libspeech. Loads the
 * model without a warm-up, compares the information read without loading with the loaded model's, refuses each
 * option's values with the category and the option's name, and checks the information an enhancement model gives: the
 * task, the sample rate 0, its options with their defaults and ranges. Then for each dump of reference/semamba/dump.py:
 * speech_enhance() of its audio at its rate and with its chunk settings passes as many samples as the audio has, its
 * progress rises to 1 and its result counts the samples, complete; the same audio given a piece at a time, in pieces of
 * random sizes, passes the same samples bit for bit; with bwe, a request and a stream pass the audio at bwe's rate, the
 * same samples. It refuses what an enhancement cannot take (another task's calls, a text, no audio, a rate outside the
 * model's range without bwe, audio shorter than its STFT, a chunk shorter than it) and runs a request it refused before
 * its work again; a callback that returns nonzero stops a request and a stream, which then takes no more audio; a stream
 * refuses a request with audio and audio too short at its end; and a request is cancelled from another thread while it
 * runs.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_DUMPS 16

/** A dump's rate and chunk settings, read from its meta.json; 0 when the file is not dump.py's. */
typedef struct {
    int rate;
    double chunk, portion;
} Meta;

static double number_after(const char * json, const char * key) {
    const char * at = strstr(json, key);
    return at ? strtod(at + strlen(key), NULL) : 0;
}

static Meta read_meta(const char * dump) {
    Meta m;
    memset(&m, 0, sizeof m);
    char path[4096 + 16];
    snprintf(path, sizeof path, "%s/meta.json", dump);
    size_t size = 0;
    char * json = read_whole(path, &size);
    if (!json) return m;
    m.rate = (int) number_after(json, "\"rate\": ");
    m.chunk = number_after(json, "\"chunk_size_in_seconds\": ");
    m.portion = number_after(json, "\"hop_length_portion\": ");
    free(json);
    return m;
}

/** An audio callback that appends to the Audio at `user_data`, and stops once it holds `stop_after` samples, when above 0. */
typedef struct {
    Audio audio;
    size_t stop_after;
} Stopping;

static int collect_until(const float * samples, size_t n, void * user_data) {
    Stopping * s = (Stopping *) user_data;
    collect(samples, n, &s->audio);
    return s->stop_after > 0 && s->audio.n >= s->stop_after;
}

/** A request on `model` with `n` samples at `rate` and the chunk settings, and bwe when above 0; a failure ends the process. */
static speech_request * enhancement_request(speech_model * model, const float * samples, size_t n, int rate, double chunk, double portion, int bwe) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        (chunk > 0 && speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, chunk) != SPEECH_OK) ||
        (portion > 0 && speech_request_set_float(r, SPEECH_OPT_HOP_LENGTH_PORTION, portion) != SPEECH_OK) ||
        (bwe > 0 && speech_request_set_int(r, SPEECH_OPT_BWE, bwe) != SPEECH_OK)) {
        fail("an enhancement request");
        exit(1);
    }
    return r;
}

/**
 * The samples a stream passes for `n` samples at `rate` given in pieces of random sizes up to `largest`, with the chunk
 * settings and bwe when above 0, into `out`; the status of the last call.
 */
static speech_status streamed(speech_model * model, const float * samples, size_t n, int rate, double chunk, double portion, int bwe, size_t largest,
                              unsigned seed, Audio * out) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return SPEECH_ERROR_INTERNAL;
    if ((chunk > 0 && speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, chunk) != SPEECH_OK) ||
        (portion > 0 && speech_request_set_float(r, SPEECH_OPT_HOP_LENGTH_PORTION, portion) != SPEECH_OK) ||
        (bwe > 0 && speech_request_set_int(r, SPEECH_OPT_BWE, bwe) != SPEECH_OK)) {
        speech_request_free(r);
        return SPEECH_ERROR_INTERNAL;
    }
    speech_enhancement * e = NULL;
    speech_status s = speech_enhancement_start(r, rate, collect, out, &e);
    speech_request_free(r);
    if (s != SPEECH_OK) return s;
    srand(seed);
    for (size_t at = 0; at < n && s == SPEECH_OK;) {
        size_t piece = 1 + (size_t) rand() % largest;
        if (piece > n - at) piece = n - at;
        s = speech_enhancement_push(e, samples + at, piece);
        at += piece;
    }
    if (s == SPEECH_OK) s = speech_enhancement_end(e);
    speech_enhancement_free(e);
    return s;
}

static int same(const Audio * a, const Audio * b) {
    return a->n == b->n && (a->n == 0 || memcmp(a->samples, b->samples, a->n * sizeof(float)) == 0);
}

/** A thread that cancels a request once the request has passed audio or reported progress. */
typedef struct {
    speech_request * request;
    Monitor monitor;
    int started, cancelled, after;
    Audio audio;
} Cancelling;

static int on_cancelling_audio(const float * samples, size_t n, void * user_data) {
    Cancelling * c = (Cancelling *) user_data;
    monitor_lock(&c->monitor);
    collect(samples, n, &c->audio);
    c->after += c->cancelled;
    c->started = 1;
    monitor_signal(&c->monitor);
    monitor_unlock(&c->monitor);
    return 0;
}

static int on_cancelling_progress(double done, void * user_data) {
    (void) done;
    Cancelling * c = (Cancelling *) user_data;
    monitor_lock(&c->monitor);
    c->after += c->cancelled;
    c->started = 1;
    monitor_signal(&c->monitor);
    monitor_unlock(&c->monitor);
    return 0;
}

static void canceller(void * arg) {
    Cancelling * c = (Cancelling *) arg;
    monitor_lock(&c->monitor);
    while (!c->started) monitor_wait(&c->monitor);
    speech_request_cancel(c->request);
    c->cancelled = 1;
    monitor_unlock(&c->monitor);
}

/** What an enhancement model refuses, and a refused request run again once it is fixed. */
static int check_refusals(speech_model * model, const char * model_path, const float * samples, size_t n, int rate) {
    speech_request * r = NULL;
    Audio audio;
    memset(&audio, 0, sizeof audio);
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_request_set_audio(r, samples, n, 0), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio without a sample rate");
    ok &= expect(speech_request_set_text(r, "明日の東京は晴れです。"), SPEECH_ERROR_UNSUPPORTED, "text", "a text for an enhancement");
    ok &= expect(speech_synthesize(r, collect, &audio), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_synthesize() of an enhancement model");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_transcribe() of an enhancement model");
    ok &= expect(speech_detect(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_detect() of an enhancement model");
    ok &= expect(speech_diarize(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_diarize() of an enhancement model");
    speech_detection * detection = NULL;
    ok &= expect(speech_detection_start(r, rate, &detection), SPEECH_ERROR_UNSUPPORTED, NULL, "a detection of an enhancement model") && detection == NULL;
    speech_diarization * diarization = NULL;
    ok &= expect(speech_diarization_start(r, rate, &diarization), SPEECH_ERROR_UNSUPPORTED, NULL, "a diarization of an enhancement model") &&
          diarization == NULL;
    ok &= expect(speech_enhance(r, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "an enhancement without a callback");
    ok &= expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "an enhancement without audio");
    ok &= expect(speech_request_set_int(r, SPEECH_OPT_BWE, 50000), SPEECH_ERROR_OUT_OF_RANGE, "bwe", "bwe above the model's rates");
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_HOP_LENGTH_PORTION, 1.5), SPEECH_ERROR_OUT_OF_RANGE, "hop_length_portion",
                 "hop_length_portion above 1");
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, 0), SPEECH_ERROR_OUT_OF_RANGE, "chunk_size_in_seconds",
                 "a chunk of 0 s");
    // Audio at a rate outside the model's range, audio shorter than its STFT and a chunk shorter than it are refused when
    // the request runs, before any audio, and the request then runs once fixed.
    ok &= expect(speech_request_set_audio(r, samples, n, 96000), SPEECH_OK, NULL, "audio at 96000 Hz") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_OUT_OF_RANGE, "audio", "audio at 96000 Hz without bwe");
    ok &= expect(speech_request_set_audio(r, samples, n, 4000), SPEECH_OK, NULL, "audio at 4000 Hz") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_OUT_OF_RANGE, "audio", "audio at 4000 Hz without bwe");
    ok &= expect(speech_request_set_audio(r, samples, 100, rate), SPEECH_OK, NULL, "100 samples") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_OUT_OF_RANGE, "audio", "audio shorter than the STFT");
    ok &= expect(speech_request_set_audio(r, samples, n, rate), SPEECH_OK, NULL, "the audio") &&
          expect(speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, 0.001), SPEECH_OK, NULL, "a chunk of 1 ms") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_OUT_OF_RANGE, "chunk_size_in_seconds", "a chunk shorter than the STFT");
    ok &= audio.n == 0;
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, 5), SPEECH_OK, NULL, "a chunk of 5 s") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_OK, NULL, "the same request run again once fixed") &&
          expect(speech_enhance(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again after its work");
    ok &= audio.n == n;
    speech_request_free(r);
    free(audio.samples);

    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    size_t tokens = 0;
    ok &= expect(speech_model_info_text_tokens(info, "明日", &tokens), SPEECH_ERROR_UNSUPPORTED, NULL, "the tokens of a text for an enhancement");
    speech_model_info_free(info);
    ok &= expect(speech_voice_add(model, "voice", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice for an enhancement");
    ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                 "a voice file for an enhancement model");
    return ok ? 0 : 1;
}

/** What a stream refuses, and a stream that its callback stops. */
static int check_stream_refusals(speech_model * model, const float * samples, size_t n, int rate) {
    Audio audio;
    memset(&audio, 0, sizeof audio);
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    speech_enhancement * e = NULL;
    int ok = expect(speech_enhancement_start(r, rate, NULL, NULL, &e), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a stream without a callback") && e == NULL;
    ok &= expect(speech_enhancement_start(r, 0, collect, &audio, &e), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a stream without a rate") && e == NULL;
    ok &= expect(speech_request_set_audio(r, samples, n, rate), SPEECH_OK, NULL, "audio of the request") &&
          expect(speech_enhancement_start(r, rate, collect, &audio, &e), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a stream of a request with audio") &&
          e == NULL;
    speech_request_free(r);
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    ok &= expect(speech_enhancement_start(r, rate, collect, &audio, &e), SPEECH_OK, NULL, "a stream") &&
          expect(speech_enhancement_push(e, samples, 100), SPEECH_OK, NULL, "100 samples") &&
          expect(speech_enhancement_end(e), SPEECH_ERROR_OUT_OF_RANGE, "audio", "a stream of audio shorter than the STFT") &&
          expect(speech_enhancement_push(e, samples, n), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a push after a failed end");
    speech_enhancement_free(e);
    e = NULL;
    // A callback that asks to stop after its first samples stops the stream, which takes no more audio.
    Stopping stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_after = 1;
    if (speech_request_set_float(r, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, 0.25) != SPEECH_OK) return fail("a chunk of 0.25 s");
    ok &= expect(speech_enhancement_start(r, rate, collect_until, &stopping, &e), SPEECH_OK, NULL, "a stream of chunks of 0.25 s");
    speech_status s = SPEECH_OK;
    for (size_t at = 0; at < n && s == SPEECH_OK; at += (size_t) rate / 10) {
        s = speech_enhancement_push(e, samples + at, at + (size_t) rate / 10 > n ? n - at : (size_t) rate / 10);
    }
    if (s == SPEECH_OK) s = speech_enhancement_end(e);
    ok &= expect(s, SPEECH_CANCELLED, NULL, "a stream whose callback asks to stop") &&
          expect(speech_enhancement_push(e, samples, n), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a push after the callback stopped the stream") &&
          expect(speech_enhancement_end(e), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "an end after the callback stopped the stream");
    speech_enhancement_free(e);
    speech_request_free(r);
    free(audio.samples);
    free(stopping.audio.samples);
    return ok ? 0 : 1;
}

/** The checks of a loaded enhancement model, which check_enhancement() frees whatever they find. */
static int check_loaded(speech_model * model, const char * model_path, const char ** dumps, int n_dumps) {
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    static const speech_option taken[] = {SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, SPEECH_OPT_HOP_LENGTH_PORTION, SPEECH_OPT_BWE};
    double chunk = 0, portion = 0, lowest = 0, highest = 0, minimum = 0, maximum = 0;
    int64_t integer = 0;
    int exclusive = 0, bwe_exclusive = 1;
    int described = speech_model_info_task(info) == SPEECH_TASK_ENHANCEMENT && speech_model_info_sample_rate(info) == 0 &&
                    !speech_model_info_incremental(info) && !speech_model_info_voice_count(info) && !speech_model_info_voice_files(info) &&
                    !speech_model_info_max_text_tokens(info) && speech_model_info_language_count(info) == 0 &&
                    speech_model_info_option_count(info) == sizeof taken / sizeof taken[0] &&
                    strstr(speech_model_info_json(info), "\"task\":\"enhancement\",\"sample_rate\":0") != NULL;
    for (size_t i = 0; described && i < sizeof taken / sizeof taken[0]; i++) described = speech_model_info_option(info, i) == taken[i];
    described = described && speech_model_info_option_default_float(info, SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, &chunk) == SPEECH_OK && chunk > 0 &&
                speech_model_info_option_default_float(info, SPEECH_OPT_HOP_LENGTH_PORTION, &portion) == SPEECH_OK && portion > 0 && portion <= 1 &&
                !speech_model_info_option_has_default(info, SPEECH_OPT_BWE) &&
                speech_model_info_option_default_int(info, SPEECH_OPT_BWE, &integer) == SPEECH_ERROR_INVALID_ARGUMENT &&
                speech_model_info_option_range(info, SPEECH_OPT_BWE, &lowest, &highest, &bwe_exclusive) == SPEECH_OK && lowest > 0 && highest >= lowest &&
                !bwe_exclusive && speech_model_info_option_range(info, SPEECH_OPT_HOP_LENGTH_PORTION, &minimum, &maximum, &exclusive) == SPEECH_OK &&
                minimum == 0 && maximum == 1 && exclusive;
    speech_model_info_free(info);
    if (!described) {
        fprintf(stderr, "FAIL: the enhancement model is not described as speech.h declares\n");
        return 1;
    }
    printf("an enhancement model of rates %g to %g Hz, chunks of %g s by default, a hop of %g of a chunk\n", lowest, highest, chunk, portion);

    int ok = 1, longest = -1;
    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    Meta metas[MAX_DUMPS];
    for (int d = 0; d < n_dumps; d++) {
        audio[d] = read_audio(dumps[d], &lengths[d]);
        metas[d] = read_meta(dumps[d]);
        if (!audio[d] || metas[d].rate <= 0 || metas[d].chunk <= 0 || metas[d].portion <= 0) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples or no meta.json of dump.py's form\n", dumps[d]);
            return 1;
        }
        const Meta m = metas[d];
        speech_request * r = enhancement_request(model, audio[d], lengths[d], m.rate, m.chunk, m.portion, 0);
        Progress progress;
        memset(&progress, 0, sizeof progress);
        speech_request_set_progress(r, record_progress, &progress);
        Audio whole;
        memset(&whole, 0, sizeof whole);
        const double start = now_seconds();
        if (speech_enhance(r, collect, &whole) != SPEECH_OK) return fail("speech_enhance");
        const double took = now_seconds() - start;
        const speech_result * result = speech_request_result(r);
        const int counted = whole.n == lengths[d] && speech_result_samples(result) == whole.n && speech_result_stop(result) == SPEECH_STOP_COMPLETE &&
                            speech_result_seed(result) == -1 && speech_result_text(result) == NULL;
        speech_request_free(r);
        Audio pieces;
        memset(&pieces, 0, sizeof pieces);
        const speech_status s = streamed(model, audio[d], lengths[d], m.rate, m.chunk, m.portion, 0, (size_t) m.rate / 3, 7, &pieces);
        const int equal = s == SPEECH_OK && same(&pieces, &whole);
        printf("%s: %zu samples at %d Hz in %.3f s, %s, %s\n", dumps[d], whole.n, m.rate, took, counted ? "every sample, complete" : "MISCOUNTED",
               equal ? "in pieces the same bit for bit" : "IN PIECES DIFFERENT");
        ok = ok && counted && equal && progress_rises(&progress, 1, "the enhancement");
        free(whole.samples);
        free(pieces.samples);
        if (longest < 0 || lengths[d] > lengths[longest]) longest = d;
    }
    if (!ok) {
        fprintf(stderr, "FAIL: an enhancement passed other samples than its audio's, or in pieces others than whole\n");
        return 1;
    }

    // bwe: the audio at another rate within the model's, from a request and from a stream alike. The longest audio has
    // chunks of 0.25 s enough for a request to stop between them.
    const float * samples = audio[longest];
    const size_t n = lengths[longest];
    const int rate = metas[longest].rate;
    const int bwe = rate != (int) lowest ? (int) lowest : (int) highest;
    speech_request * r = enhancement_request(model, samples, n, rate, 0, 0, bwe);
    Audio resampled, pieces;
    memset(&resampled, 0, sizeof resampled);
    memset(&pieces, 0, sizeof pieces);
    if (speech_enhance(r, collect, &resampled) != SPEECH_OK) return fail("speech_enhance with bwe");
    speech_request_free(r);
    const double expected = (double) n * bwe / rate;
    const speech_status s = streamed(model, samples, n, rate, 0, 0, bwe, (size_t) rate / 3, 9, &pieces);
    if (fabs((double) resampled.n - expected) > 1 || s != SPEECH_OK || !same(&pieces, &resampled)) {
        fprintf(stderr, "FAIL: bwe %d gave %zu samples where %.1f were expected, or a stream other samples (%zu)\n", bwe, resampled.n, expected, pieces.n);
        return 1;
    }
    printf("bwe %d: %zu samples from %zu at %d Hz, a stream the same\n", bwe, resampled.n, n, rate);
    free(resampled.samples);
    free(pieces.samples);

    if (check_refusals(model, model_path, samples, n, rate) != 0 || check_stream_refusals(model, samples, n, rate) != 0) return 1;

    // A callback that returns nonzero stops a request after its first samples.
    r = enhancement_request(model, samples, n, rate, 0.25, 0, 0);
    Stopping stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_after = 1;
    const speech_status stopped = speech_enhance(r, collect_until, &stopping);
    const speech_result * result = speech_request_result(r);
    if (!expect(stopped, SPEECH_CANCELLED, NULL, "a callback that asks to stop") || !result || speech_result_stop(result) != SPEECH_STOP_CANCELLED ||
        speech_result_samples(result) != stopping.audio.n || stopping.audio.n >= n) {
        fprintf(stderr, "FAIL: a stopped enhancement's result does not count the samples it passed\n");
        return 1;
    }
    printf("a callback that asks to stop: cancelled after %zu of %zu samples\n", stopping.audio.n, n);
    speech_request_free(r);
    free(stopping.audio.samples);

    // Another thread cancels a request of chunks of 0.25 s once it has passed audio or reported progress.
    Cancelling c;
    memset(&c, 0, sizeof c);
    monitor_init(&c.monitor);
    c.request = enhancement_request(model, samples, n, rate, 0.25, 0, 0);
    speech_request_set_progress(c.request, on_cancelling_progress, &c);
    Thread thread;
    thread_start(&thread, canceller, &c);
    const speech_status status = speech_enhance(c.request, on_cancelling_audio, &c);
    thread_join(&thread);
    result = speech_request_result(c.request);
    if (status == SPEECH_CANCELLED && result && speech_result_stop(result) == SPEECH_STOP_CANCELLED && c.after <= 1) {
        printf("cancelled from another thread: %zu samples, %d call(s) of a callback after the cancel\n", c.audio.n, c.after);
    } else if (status == SPEECH_OK && c.audio.n == n) {
        printf("the request finished before the cancel took effect; cancellation was not exercised\n");
    } else {
        fprintf(stderr, "FAIL: a cancel from another thread did not stop the request (status %d)\n", (int) status);
        return 1;
    }
    speech_request_free(c.request);
    monitor_free(&c.monitor);
    free(c.audio.samples);
    for (int d = 0; d < n_dumps; d++) free(audio[d]);
    return 0;
}

int check_enhancement(int argc, char ** argv) {
    const char * model_path = argv[0];
    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    const char * dumps[MAX_DUMPS];
    int n_dumps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
        } else if (argv[i][0] != '-' && n_dumps < MAX_DUMPS) {
            dumps[n_dumps++] = argv[i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (n_dumps == 0) {
        fprintf(stderr, "give at least one dump folder of reference/semamba/dump.py\n");
        return 2;
    }
    speech_model * model = NULL;
    const double load_start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - load_start);
    const int status = check_loaded(model, model_path, dumps, n_dumps);
    speech_model_free(model);
    if (status == 0) printf("ok\n");
    return status;
}
