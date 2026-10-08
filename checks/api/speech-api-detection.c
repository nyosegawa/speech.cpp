/*
 * The detection part of speech-api-check: the C API with a detection model, through the shared libspeech. Loads the
 * model without a warm-up, compares the information read without loading with the loaded model's, refuses each
 * option's values with the category and the option's name, finds the regions of each dump of
 * reference/silero-vad/dump.py with every set of options in its regions.json and compares them with the official's
 * sample for sample, finds regions in audio at three times the model's rate, refuses what a detection cannot take and
 * runs a request it refused before its work again, reports and stops on progress, and cancels a request from another
 * thread while it runs, its result then without regions. Then the detections of audio given a piece at a time,
 * speech-api-detection-stream.c.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_DUMPS 16
#define MAX_SETS 16
#define MAX_OPTIONS 8
#define MAX_REGIONS 256

/** One set of options of a dump's regions.json and the regions the official gave with it, in samples. */
typedef struct {
    char name[64];
    int n_options;
    char option[MAX_OPTIONS][64];
    double value[MAX_OPTIONS];
    int n_regions;
    long long start[MAX_REGIONS], end[MAX_REGIONS];
} Set;

/**
 * The sets of a regions.json as dump.py writes it, {"name": {"options": {"key": number, ...}, "regions": [[start, end],
 * ...]}, ...}, read into `sets`; the number read, or -1 when the text is not of that form.
 */
static int read_sets(const char * json, Set * sets) {
    const char * at = strchr(json, '{');
    int n = 0;
    while (at && n < MAX_SETS) {
        const char * name = strchr(at + 1, '"');
        if (!name) break;
        const char * name_end = strchr(name + 1, '"');
        const char * options = strstr(name, "\"options\": {"), * regions = strstr(name, "\"regions\": [");
        if (!name_end || !options || !regions || (size_t) (name_end - name - 1) >= sizeof sets[n].name) return -1;
        Set * s = &sets[n];
        memset(s, 0, sizeof *s);
        memcpy(s->name, name + 1, (size_t) (name_end - name - 1));
        const char * p = options + strlen("\"options\": {");
        while (*p && *p != '}') {
            const char * key = strchr(p, '"');
            if (!key || key > strchr(p, '}')) break;
            const char * key_end = strchr(key + 1, '"');
            if (!key_end || s->n_options >= MAX_OPTIONS || (size_t) (key_end - key - 1) >= sizeof s->option[0]) return -1;
            memcpy(s->option[s->n_options], key + 1, (size_t) (key_end - key - 1));
            char * number_end = NULL;
            s->value[s->n_options++] = strtod(key_end + 2, &number_end);
            p = number_end;
        }
        p = regions + strlen("\"regions\": [");
        for (;;) {
            while (*p == ' ' || *p == '\n' || *p == ',') p++;
            if (*p == ']') break;
            if (*p != '[' || s->n_regions >= MAX_REGIONS) return -1;
            char * next = NULL;
            s->start[s->n_regions] = strtoll(p + 1, &next, 10);
            while (*next == ',' || *next == ' ' || *next == '\n') next++;
            s->end[s->n_regions++] = strtoll(next, &next, 10);
            p = strchr(next, ']');
            if (!p) return -1;
            p++;
        }
        n++;
        at = strchr(p + 1, '}');
        at = at && strchr(at, '"') ? at : NULL;
    }
    return n;
}

/** Sets an option of a set through the setter of its type. */
static speech_status set_option(speech_request * r, const char * name, double value) {
    speech_option option;
    const speech_status s = speech_option_from_name(name, &option);
    if (s != SPEECH_OK) return s;
    return speech_option_type(option) == SPEECH_TYPE_INT ? speech_request_set_int(r, option, (int64_t) value) : speech_request_set_float(r, option, value);
}

/**
 * Finds the regions of `n` samples at `rate` with the options of `set` (NULL for none), returning the status, and the
 * regions in samples at the model's rate `model_rate` in `starts` and `ends`, of which there are `*count`, after checking
 * that the result is a detection's.
 */
static speech_status detect(speech_model * model, const float * samples, size_t n, int rate, const Set * set, Progress * progress, int model_rate,
                            long long * starts, long long * ends, int * count) {
    speech_request * r = NULL;
    *count = 0;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        (progress && speech_request_set_progress(r, record_progress, progress) != SPEECH_OK)) {
        fail("a request's audio");
        speech_request_free(r);
        return SPEECH_ERROR_INTERNAL;
    }
    for (int i = 0; set && i < set->n_options; i++) {
        if (set_option(r, set->option[i], set->value[i]) != SPEECH_OK) {
            fail(set->option[i]);
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    const speech_status status = speech_detect(r);
    const speech_result * result = speech_request_result(r);
    if (result) {
        int ok = speech_result_text(result) == NULL && speech_result_token_count(result) == 0 && speech_result_language_count(result) == 0 &&
                 speech_result_seed(result) == -1 && speech_result_samples(result) == 0 &&
                 speech_result_stop(result) == (status == SPEECH_CANCELLED ? SPEECH_STOP_CANCELLED : SPEECH_STOP_COMPLETE) &&
                 speech_result_segment_count(result) <= MAX_REGIONS;
        double last = 0;
        for (size_t i = 0; ok && i < speech_result_segment_count(result); i++) {
            double start = 0, end = 0;
            const char * text = NULL;
            ok = speech_result_segment(result, i, &start, &end, &text) == SPEECH_OK && text && text[0] == '\0' && start >= last && end > start &&
                 end <= (double) n / rate + 1e-9;
            starts[i] = llround(start * model_rate);
            ends[i] = llround(end * model_rate);
            last = end;
        }
        *count = (int) speech_result_segment_count(result);
        if (!ok || speech_result_segment(result, speech_result_segment_count(result), NULL, NULL, NULL) != SPEECH_ERROR_INVALID_ARGUMENT) {
            fprintf(stderr, "FAIL: a detection's result has a text, tokens, languages, a seed or samples, or regions out of order or with text\n");
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    speech_request_free(r);
    return status;
}

static int ignore_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    (void) user_data;
    return 0;
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

/** What a detection model refuses: the wrong input and calls, and voices. */
static int check_refusals(speech_model * model, const char * model_path, const float * samples, size_t n, int rate) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_request_set_audio(r, samples, n, 0), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio without a sample rate");
    ok &= expect(speech_request_set_audio(r, samples, 0, rate), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "no audio");
    ok &= expect(speech_request_set_text(r, "明日の東京は晴れです。"), SPEECH_ERROR_UNSUPPORTED, "text", "a text for a detection");
    ok &= expect(speech_synthesize(r, ignore_audio, NULL), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_synthesize() of a detection model");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_transcribe() of a detection model");
    ok &= expect(speech_detect(r), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a detection without audio");
    // A request refused before its work is fixed and run again; one that has done its work runs once.
    ok &= expect(speech_request_set_audio(r, samples, n, rate), SPEECH_OK, NULL, "the audio") &&
          expect(speech_detect(r), SPEECH_OK, NULL, "the same request run again with its audio") &&
          expect(speech_detect(r), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again after its work");
    speech_request_free(r);
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    size_t tokens = 0;
    ok &= expect(speech_model_info_text_tokens(info, "明日", &tokens), SPEECH_ERROR_UNSUPPORTED, NULL, "the tokens of a text for a detection");
    speech_model_info_free(info);
    ok &= expect(speech_voice_add(model, "voice", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice for a detection");
    ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                 "a voice file for a detection model");
    return ok ? 0 : 1;
}

/** The checks of a loaded detection model, which check_detection() frees whatever they find. */
static int check_loaded(speech_model * model, const char * model_path, const char ** dumps, int n_dumps) {
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info);
    static const speech_option taken[] = {SPEECH_OPT_THRESHOLD, SPEECH_OPT_MIN_SPEECH_DURATION_MS, SPEECH_OPT_MIN_SILENCE_DURATION_MS,
                                          SPEECH_OPT_SPEECH_PAD_MS, SPEECH_OPT_MAX_SPEECH_DURATION_S};
    int described = speech_model_info_task(info) == SPEECH_TASK_DETECTION && !speech_model_info_incremental(info) &&
                    !speech_model_info_voice_count(info) && !speech_model_info_voice_files(info) && !speech_model_info_max_text_tokens(info) &&
                    rate > 0 && speech_model_info_language_count(info) == 0 && speech_model_info_language(info, 0) == NULL &&
                    speech_model_info_option_count(info) == sizeof taken / sizeof taken[0] && strstr(speech_model_info_json(info), "\"task\":\"detection\"");
    for (size_t i = 0; described && i < sizeof taken / sizeof taken[0]; i++) described = speech_model_info_option(info, i) == taken[i];
    speech_model_info_free(info);
    if (!described) {
        fprintf(stderr, "FAIL: the detection model is not described as speech.h declares\n");
        return 1;
    }

    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    static Set sets[MAX_SETS];
    static long long starts[MAX_REGIONS], ends[MAX_REGIONS];
    int longest = 0, with_speech = -1, ok = 1;
    for (int d = 0; d < n_dumps; d++) {
        char path[4096 + 16];
        size_t size = 0;
        snprintf(path, sizeof path, "%s/regions.json", dumps[d]);
        char * json = read_whole(path, &size);
        audio[d] = read_audio(dumps[d], &lengths[d]);
        const int n_sets = json ? read_sets(json, sets) : -1;
        free(json);
        if (!audio[d] || n_sets <= 0) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples or no regions.json of dump.py's form\n", dumps[d]);
            return 1;
        }
        for (int k = 0; k < n_sets; k++) {
            const Set * set = &sets[k];
            Progress progress;
            memset(&progress, 0, sizeof progress);
            int count = 0;
            const double start = now_seconds();
            if (detect(model, audio[d], lengths[d], rate, set, &progress, rate, starts, ends, &count) != SPEECH_OK) return fail("speech_detect");
            int equal = count == set->n_regions;
            for (int i = 0; equal && i < count; i++) equal = starts[i] == set->start[i] && ends[i] == set->end[i];
            printf("%s %s: %.2f s of audio in %.3f s, %d regions, %s\n", dumps[d], set->name, (double) lengths[d] / rate, now_seconds() - start, count,
                   equal ? "the official's" : "DIFFERENT FROM THE OFFICIAL'S");
            ok = ok && equal && progress_rises(&progress, 1, "the detection");
            if (!strcmp(set->name, "default") && set->n_regions > 0 && (with_speech < 0 || lengths[d] < lengths[with_speech])) with_speech = d;
        }
        if (lengths[d] > lengths[longest]) longest = d;
    }
    if (!ok || with_speech < 0) {
        fprintf(stderr, "FAIL: regions differ from the official's, the progress does not rise, or no dump has speech\n");
        return 1;
    }

    // Each sample three times is audio at three times the rate, which the library resamples to the model's.
    const size_t n = lengths[with_speech];
    float * tripled = (float *) malloc(3 * n * sizeof(float));
    for (size_t i = 0; i < 3 * n; i++) tripled[i] = audio[with_speech][i / 3];
    int count = 0;
    if (detect(model, tripled, 3 * n, 3 * rate, NULL, NULL, rate, starts, ends, &count) != SPEECH_OK || count == 0) {
        return fail("audio at three times the rate");
    }
    printf("the same audio at %d Hz: %d regions, the first from %.3f s to %.3f s\n", 3 * rate, count, (double) starts[0] / rate, (double) ends[0] / rate);
    free(tripled);
    if (check_refusals(model, model_path, audio[with_speech], n, rate) != 0) return 1;
    if (check_detection_stream(model, audio, lengths, dumps, n_dumps) != 0) return 1;

    Progress stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_at = 0.5;
    if (!expect(detect(model, audio[longest], lengths[longest], rate, NULL, &stopping, rate, starts, ends, &count), SPEECH_CANCELLED, NULL,
                "a progress callback that stops at 0.5") ||
        count != 0) {
        fprintf(stderr, "FAIL: a stopped detection has regions\n");
        return 1;
    }

    // The cancel comes 5 ms into the longest audio, within its first block of chunks.
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, audio[longest], lengths[longest], rate) != SPEECH_OK) {
        return fail("a request");
    }
    Canceller c = {r, 0.005, 0};
    Thread thread;
    thread_start(&thread, cancel_later, &c);
    const double start = now_seconds();
    const speech_status status = speech_detect(r);
    const double took = now_seconds() - start;
    thread_join(&thread);
    const speech_result * result = speech_request_result(r);
    if (status == SPEECH_CANCELLED && result && speech_result_segment_count(result) == 0 && speech_result_stop(result) == SPEECH_STOP_CANCELLED) {
        printf("cancelled from another thread after %.3f s: stopped at %.3f s without regions\n", c.cancelled_at, took);
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

int check_detection(int argc, char ** argv) {
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
        fprintf(stderr, "give at least one dump folder of reference/silero-vad/dump.py\n");
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
