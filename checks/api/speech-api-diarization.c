/*
 * The diarization part of speech-api-check: the C API with a diarization model, through the shared libspeech. Loads the
 * model without a warm-up, compares the information read without loading with the loaded model's, refuses each
 * option's values with the category and the option's name, diarizes the audio of each dump of
 * reference/sortformer/dump.py (an input's folder, which holds audio.npy) and checks its segments' order, speakers and
 * times; then gives the same audio a piece at a time to diarizations, which must give speech_diarize()'s segments in each
 * streaming mode, none before it is final, while two run at once on the model, each numbering its own speakers; and
 * diarizes audio at three times the model's rate, refuses what a diarization cannot take, runs a request it refused
 * before its work again, reports and stops on progress, and cancels a request from another thread while it runs, its
 * result then without segments.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_DUMPS 16
#define MAX_SEGMENTS 1024

typedef struct {
    double start, end;
    int speaker;
} Segment;

/** Segments ordered by start, then by speaker, then by end, as speech_diarize() gives them. */
static int before(const Segment * a, const Segment * b) {
    if (a->start != b->start) return a->start < b->start;
    if (a->speaker != b->speaker) return a->speaker < b->speaker;
    return a->end < b->end;
}

static int compare_segments(const void * a, const void * b) {
    return before((const Segment *) a, (const Segment *) b) ? -1 : before((const Segment *) b, (const Segment *) a) ? 1 : 0;
}

/**
 * Diarizes `n` samples at `rate` with `mode` (NULL for the default) on `model`, returning the status, and the segments in
 * `out`, of which there are `*count`, after checking that the result is a diarization's: no text, tokens, languages,
 * seed or samples, its segments in order, of speakers below `speakers`, without text and within the audio.
 */
static speech_status diarize(speech_model * model, const float * samples, size_t n, int rate, const char * mode, Progress * progress, int speakers,
                             Segment * out, int * count) {
    speech_request * r = NULL;
    *count = 0;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        (mode && speech_request_set_string(r, SPEECH_OPT_STREAMING_MODE, mode) != SPEECH_OK) ||
        (progress && speech_request_set_progress(r, record_progress, progress) != SPEECH_OK)) {
        fail("a request's audio and streaming mode");
        speech_request_free(r);
        return SPEECH_ERROR_INTERNAL;
    }
    const speech_status status = speech_diarize(r);
    const speech_result * result = speech_request_result(r);
    if (result) {
        const size_t found = speech_result_segment_count(result);
        int ok = speech_result_text(result) == NULL && speech_result_token_count(result) == 0 && speech_result_language_count(result) == 0 &&
                 speech_result_seed(result) == -1 && speech_result_samples(result) == 0 &&
                 speech_result_stop(result) == (status == SPEECH_CANCELLED ? SPEECH_STOP_CANCELLED : SPEECH_STOP_COMPLETE) && found <= MAX_SEGMENTS &&
                 speech_result_segment_speaker(result, found) == -1;
        for (size_t i = 0; ok && i < found; i++) {
            const char * text = NULL;
            Segment * s = &out[i];
            ok = speech_result_segment(result, i, &s->start, &s->end, &text) == SPEECH_OK && text && text[0] == '\0' && s->end > s->start &&
                 s->start >= 0 && s->end <= (double) n / rate + 0.011;
            s->speaker = speech_result_segment_speaker(result, i);
            ok = ok && s->speaker >= 0 && s->speaker < speakers && (i == 0 || !before(s, &out[i - 1]));
        }
        *count = (int) found;
        if (!ok || speech_result_segment(result, found, NULL, NULL, NULL) != SPEECH_ERROR_INVALID_ARGUMENT) {
            fprintf(stderr, "FAIL: a diarization's result has a text, tokens, languages, a seed or samples, or segments out of order, with text or of "
                            "another speaker\n");
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    speech_request_free(r);
    return status;
}

/** A diarization of audio given a piece at a time, and what it has given so far. */
typedef struct {
    speech_diarization * diarization;
    const float * samples;
    size_t n, at, piece;
    double settled;
    int count, early;
    Segment segments[MAX_SEGMENTS];
} Stream;

/** Starts a diarization of `n` samples at `rate` in `mode` (NULL for the default), given in pieces of `piece`. */
static int start_stream(Stream * s, speech_model * model, const float * samples, size_t n, int rate, const char * mode, size_t piece) {
    memset(s, 0, sizeof *s);
    s->samples = samples;
    s->n = n;
    s->piece = piece;
    speech_request * r = NULL;
    int ok = speech_request_new(model, &r) == SPEECH_OK && (!mode || speech_request_set_string(r, SPEECH_OPT_STREAMING_MODE, mode) == SPEECH_OK) &&
             speech_diarization_start(r, rate, &s->diarization) == SPEECH_OK;
    speech_request_free(r);
    return ok ? 0 : fail("speech_diarization_start");
}

/**
 * Takes the segments given since the last call, each of which must end before the settled time it was given at, and the
 * settled time, which must not fall; `*speakers` is the model's.
 */
static int take_given(Stream * s, int speakers) {
    const double settled = speech_diarization_settled(s->diarization);
    if (settled < s->settled) return fail("the settled time fell");
    s->settled = settled;
    for (size_t i = (size_t) s->count; i < speech_diarization_segment_count(s->diarization); i++) {
        if (s->count >= MAX_SEGMENTS) return fail("more segments than the check holds");
        Segment * g = &s->segments[s->count++];
        if (speech_diarization_segment(s->diarization, i, &g->start, &g->end, &g->speaker) != SPEECH_OK) return fail("speech_diarization_segment");
        if (g->end > settled + 0.011 || g->speaker < 0 || g->speaker >= speakers) s->early++;
    }
    for (int k = 0; k < speakers; k++) {
        double start = 0;
        if (speech_diarization_speaking(s->diarization, k, &start) && start > settled + 0.011) s->early++;
    }
    return 0;
}

/** Gives the stream its next piece, or ends it after the last; 1 once it has ended. */
static int step_stream(Stream * s, int speakers, int * failed) {
    if (s->at >= s->n) {
        if (speech_diarization_end(s->diarization) != SPEECH_OK) *failed |= fail("speech_diarization_end");
        *failed |= take_given(s, speakers);
        return 1;
    }
    const size_t k = s->n - s->at < s->piece ? s->n - s->at : s->piece;
    if (speech_diarization_push(s->diarization, s->samples + s->at, k) != SPEECH_OK) *failed |= fail("speech_diarization_push");
    s->at += k;
    *failed |= take_given(s, speakers);
    return 0;
}

/** Whether the stream gave `count` segments, `want`, as a set, none of them early. */
static int same_as(Stream * s, const Segment * want, int count) {
    qsort(s->segments, (size_t) s->count, sizeof(Segment), compare_segments);
    int same = s->count == count && s->early == 0;
    for (int i = 0; same && i < count; i++) {
        same = s->segments[i].speaker == want[i].speaker && s->segments[i].start == want[i].start && s->segments[i].end == want[i].end;
    }
    return same;
}

static int ignore_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    (void) user_data;
    return 0;
}

/** What a diarization model refuses: the wrong input and calls, and voices. */
static int check_refusals(speech_model * model, const char * model_path, const float * samples, size_t n, int rate) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_request_set_audio(r, samples, n, 0), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio without a sample rate");
    ok &= expect(speech_request_set_text(r, "明日の東京は晴れです。"), SPEECH_ERROR_UNSUPPORTED, "text", "a text for a diarization");
    ok &= expect(speech_synthesize(r, ignore_audio, NULL), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_synthesize() of a diarization model");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_transcribe() of a diarization model");
    ok &= expect(speech_detect(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_detect() of a diarization model");
    speech_detection * detection = NULL;
    ok &= expect(speech_detection_start(r, rate, &detection), SPEECH_ERROR_UNSUPPORTED, NULL, "a detection of a diarization model") && detection == NULL;
    ok &= expect(speech_diarize(r), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a diarization without audio");
    speech_diarization * diarization = NULL;
    ok &= expect(speech_diarization_start(r, 0, &diarization), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a diarization at no rate") && diarization == NULL;
    // A request refused before its work is fixed and run again; one that has done its work runs once.
    ok &= expect(speech_request_set_audio(r, samples, n, rate), SPEECH_OK, NULL, "the audio") &&
          expect(speech_diarization_start(r, rate, &diarization), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a diarization of a request with audio") &&
          expect(speech_diarize(r), SPEECH_OK, NULL, "the same request run again with its audio") &&
          expect(speech_diarize(r), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again after its work");
    speech_request_free(r);
    // A diarization that has ended takes no more audio, and gives no segment past its last nor a speaker it does not have.
    if (speech_request_new(model, &r) != SPEECH_OK || speech_diarization_start(r, rate, &diarization) != SPEECH_OK) return fail("a diarization");
    speech_request_free(r);
    ok &= expect(speech_diarization_push(diarization, NULL, 0), SPEECH_OK, NULL, "a push of no samples") &&
          expect(speech_diarization_push(diarization, NULL, 1), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a push of NULL samples") &&
          expect(speech_diarization_end(diarization), SPEECH_OK, NULL, "the end of no audio") &&
          expect(speech_diarization_push(diarization, samples, 1), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a push after the end") &&
          expect(speech_diarization_end(diarization), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a second end") &&
          expect(speech_diarization_segment(diarization, 0, NULL, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a segment of no audio") &&
          speech_diarization_segment_count(diarization) == 0 && isinf(speech_diarization_settled(diarization)) &&
          !speech_diarization_speaking(diarization, -1, NULL) && !speech_diarization_speaking(diarization, 0, NULL);
    speech_diarization_free(diarization);
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    size_t tokens = 0;
    ok &= expect(speech_model_info_text_tokens(info, "明日", &tokens), SPEECH_ERROR_UNSUPPORTED, NULL, "the tokens of a text for a diarization");
    speech_model_info_free(info);
    ok &= expect(speech_voice_add(model, "voice", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice for a diarization");
    ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                 "a voice file for a diarization model");
    return ok ? 0 : 1;
}

/** A thread that cancels a request a fixed time after it starts, and notes when it did. */
typedef struct {
    speech_request * request;
    double delay, cancelled_at;
} Canceller;

static void cancel_later(void * arg) {
    Canceller * c = (Canceller *) arg;
    const double start = now_seconds();
    sleep_seconds(c->delay);
    speech_request_cancel(c->request);
    c->cancelled_at = now_seconds() - start;
}

/** The checks of a loaded diarization model, which check_diarization() frees whatever they find. */
static int check_loaded(speech_model * model, const char * model_path, const char ** dumps, int n_dumps) {
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info), speakers = speech_model_info_speakers(info);
    static const speech_option taken[] = {SPEECH_OPT_STREAMING_MODE, SPEECH_OPT_SPKCACHE_LEN, SPEECH_OPT_FIFO_LEN, SPEECH_OPT_CHUNK_LEN,
                                          SPEECH_OPT_CHUNK_RIGHT_CONTEXT, SPEECH_OPT_SPKCACHE_UPDATE_PERIOD, SPEECH_OPT_ONSET, SPEECH_OPT_OFFSET,
                                          SPEECH_OPT_PAD_ONSET, SPEECH_OPT_PAD_OFFSET, SPEECH_OPT_MIN_DURATION_ON, SPEECH_OPT_MIN_DURATION_OFF};
    int described = speech_model_info_task(info) == SPEECH_TASK_DIARIZATION && speakers > 0 && !speech_model_info_incremental(info) &&
                    !speech_model_info_voice_count(info) && !speech_model_info_voice_files(info) && !speech_model_info_max_text_tokens(info) &&
                    rate > 0 && speech_model_info_language_count(info) == 0 && speech_model_info_option_count(info) == sizeof taken / sizeof taken[0] &&
                    !speech_model_info_option_has_default(info, SPEECH_OPT_STREAMING_MODE) &&
                    strstr(speech_model_info_json(info), "\"task\":\"diarization\"") && strstr(speech_model_info_json(info), "\"speakers\":");
    for (size_t i = 0; described && i < sizeof taken / sizeof taken[0]; i++) described = speech_model_info_option(info, i) == taken[i];
    const size_t n_modes = speech_model_info_option_choice_count(info, SPEECH_OPT_STREAMING_MODE);
    const char * modes[8];
    for (size_t i = 0; i < n_modes && i < 8; i++) modes[i] = speech_model_info_option_choice(info, SPEECH_OPT_STREAMING_MODE, i);
    described = described && n_modes >= 2 && n_modes <= 8;
    speech_model_info_free(info);
    if (!described) {
        fprintf(stderr, "FAIL: the diarization model is not described as speech.h declares\n");
        return 1;
    }

    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    static Segment whole[MAX_SEGMENTS];
    static Stream a, b;
    int longest = 0, with_speech = -1, ok = 1;
    for (int d = 0; d < n_dumps; d++) {
        audio[d] = read_audio(dumps[d], &lengths[d]);
        if (!audio[d]) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples\n", dumps[d]);
            return 1;
        }
        Progress progress;
        memset(&progress, 0, sizeof progress);
        int count = 0;
        const double start = now_seconds();
        if (diarize(model, audio[d], lengths[d], rate, NULL, &progress, speakers, whole, &count) != SPEECH_OK) return fail("speech_diarize");
        printf("%s: %.2f s of audio in %.3f s, %d segments\n", dumps[d], (double) lengths[d] / rate, now_seconds() - start, count);
        ok = ok && progress_rises(&progress, 1, "the diarization");
        if (count > 0 && (with_speech < 0 || lengths[d] < lengths[with_speech])) with_speech = d;
        if (lengths[d] > lengths[longest]) longest = d;
    }
    if (!ok || with_speech < 0) {
        fprintf(stderr, "FAIL: the progress does not rise, or no dump has speech\n");
        return 1;
    }

    // Each streaming mode: the longest audio and the shortest with speech, given at once in pieces of 10 ms and of a
    // prime number of samples, each stream as speech_diarize() in the mode, the default mode as low_latency.
    for (size_t m = 0; m <= n_modes; m++) {
        const char * mode = m < n_modes ? modes[m] : NULL;
        static Segment want_a[MAX_SEGMENTS], want_b[MAX_SEGMENTS];
        int count_a = 0, count_b = 0;
        if (diarize(model, audio[longest], lengths[longest], rate, mode ? mode : "low_latency", NULL, speakers, want_a, &count_a) != SPEECH_OK ||
            diarize(model, audio[with_speech], lengths[with_speech], rate, mode ? mode : "low_latency", NULL, speakers, want_b, &count_b) != SPEECH_OK) {
            return fail("speech_diarize in a streaming mode");
        }
        if (start_stream(&a, model, audio[longest], lengths[longest], rate, mode, (size_t) rate / 100) != 0 ||
            start_stream(&b, model, audio[with_speech], lengths[with_speech], rate, mode, 1777) != 0) {
            return 1;
        }
        int failed = 0, done_a = 0, done_b = 0;
        while (!done_a || !done_b) {
            if (!done_a) done_a = step_stream(&a, speakers, &failed);
            if (!done_b) done_b = step_stream(&b, speakers, &failed);
        }
        const int same = !failed && same_as(&a, want_a, count_a) && same_as(&b, want_b, count_b);
        printf("%s: two diarizations at once in pieces gave %d and %d segments, %s\n", mode ? mode : "the default mode", a.count, b.count,
               same ? "each speech_diarize()'s, none before it was final" : "DIFFERENT OR EARLY");
        speech_diarization_free(a.diarization);
        speech_diarization_free(b.diarization);
        if (!same) return 1;
    }

    // Each sample three times is audio at three times the rate, which the library resamples to the model's.
    const size_t n = lengths[with_speech];
    float * tripled = (float *) malloc(3 * n * sizeof(float));
    for (size_t i = 0; i < 3 * n; i++) tripled[i] = audio[with_speech][i / 3];
    int count = 0;
    if (diarize(model, tripled, 3 * n, 3 * rate, NULL, NULL, speakers, whole, &count) != SPEECH_OK || count == 0) {
        return fail("audio at three times the rate");
    }
    printf("the same audio at %d Hz: %d segments, the first of speaker %d from %.2f s to %.2f s\n", 3 * rate, count, whole[0].speaker, whole[0].start,
           whole[0].end);
    free(tripled);
    if (check_refusals(model, model_path, audio[with_speech], n, rate) != 0) return 1;

    // A progress callback that stops halfway, in the mode of the most steps.
    Progress stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_at = 0.5;
    if (!expect(diarize(model, audio[longest], lengths[longest], rate, modes[n_modes - 1], &stopping, speakers, whole, &count), SPEECH_CANCELLED, NULL,
                "a progress callback that stops at 0.5") ||
        count != 0) {
        fprintf(stderr, "FAIL: a stopped diarization has segments\n");
        return 1;
    }

    // The cancel comes 5 ms into the longest audio in the mode of the most steps, within its first steps.
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, audio[longest], lengths[longest], rate) != SPEECH_OK ||
        speech_request_set_string(r, SPEECH_OPT_STREAMING_MODE, modes[n_modes - 1]) != SPEECH_OK) {
        return fail("a request");
    }
    Canceller c = {r, 0.005, 0};
    Thread thread;
    thread_start(&thread, cancel_later, &c);
    const double start = now_seconds();
    const speech_status status = speech_diarize(r);
    const double took = now_seconds() - start;
    thread_join(&thread);
    const speech_result * result = speech_request_result(r);
    if (status == SPEECH_CANCELLED && result && speech_result_segment_count(result) == 0 && speech_result_stop(result) == SPEECH_STOP_CANCELLED) {
        printf("cancelled from another thread after %.3f s: stopped at %.3f s without segments\n", c.cancelled_at, took);
    } else if (status == SPEECH_OK && took < c.cancelled_at + 0.01) {
        printf("the request finished in %.3f s, before the cancel at %.3f s took effect; cancellation was not exercised\n", took, c.cancelled_at);
    } else {
        fprintf(stderr, "FAIL: a cancel at %.3f s did not stop a request that ran %.3f s (status %d)\n", c.cancelled_at, took, (int) status);
        return 1;
    }
    speech_request_free(r);
    for (int d = 0; d < n_dumps; d++) free(audio[d]);
    return 0;
}

int check_diarization(int argc, char ** argv) {
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
        fprintf(stderr, "give at least one input's folder of reference/sortformer/dump.py\n");
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
