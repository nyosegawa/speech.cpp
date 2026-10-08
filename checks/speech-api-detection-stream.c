/*
 * The detections of audio given a piece at a time, part of speech-api-check's detect mode. Each dump's audio, at the
 * model's rate and at 24 kHz and 48 kHz made from it by linear interpolation, is pushed in pieces of one sample, 20 ms,
 * 100 ms and 1 s and of random sizes from a fixed seed, with sets of options that include max_speech_duration_s and
 * min_silence_duration_ms below twice speech_pad_ms, and must give, once ended, the regions that speech_detect() gives
 * for the same audio, sample for sample. While the audio arrives, a region once given must not change; with
 * min_speech_duration_ms 0, every start the detection says someone speaks from must be a region's; and with pieces of one
 * sample, min_silence_duration_ms of at least twice speech_pad_ms and no max_speech_duration_s, every region given before
 * the end must have had its start said before it was given. Two detections on one model pushed in turn must each give
 * their own regions. Then it times a minute of audio in pieces of 20 ms, 100 ms and 1 s, and checks what a detection
 * refuses.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_REGIONS 256
#define MAX_SAID 4096

/** A set of options; NAN or a negative number leaves an option at the model's default. */
typedef struct {
    const char * name;
    double threshold;
    int64_t min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms;
    double max_speech_duration_s;
} Options;

/** Sets of dump.py's, and sets whose regions' ends wait on what follows them. */
static const Options SETS[] = {
    {"default", NAN, -1, -1, -1, NAN},
    {"strict", 0.7, 500, 300, 100, NAN},
    {"loose", 0.3, 0, 0, 0, NAN},
    {"max3", NAN, -1, -1, -1, 3},
    {"max10-silence1000", NAN, -1, 1000, -1, 10},
    {"pad100-silence0", NAN, -1, 0, 100, NAN},
    {"pad200-silence50-speech0", NAN, 0, 50, 200, NAN},
    {"pad100-silence0-max2", NAN, -1, 0, 100, 2},
};
#define N_SETS (int) (sizeof SETS / sizeof SETS[0])

static const char * const PIECES[] = {"1 sample", "20 ms", "100 ms", "1 s", "random sizes"};
#define N_PIECES 5

/** The size of the next piece of audio at `rate` of the kind PIECES[kind] names, drawing from `seed` for random sizes. */
static size_t piece(int kind, int rate, uint32_t * seed) {
    switch (kind) {
        case 0: return 1;
        case 1: return (size_t) rate / 50;
        case 2: return (size_t) rate / 10;
        case 3: return (size_t) rate;
        default:
            *seed = *seed * 1664525u + 1013904223u;
            return 1 + (size_t) (*seed >> 8) % ((size_t) rate / 5);
    }
}

static speech_status set_options(speech_request * r, const Options * o) {
    speech_status s = SPEECH_OK;
    if (!isnan(o->threshold)) s = speech_request_set_float(r, SPEECH_OPT_THRESHOLD, o->threshold);
    if (s == SPEECH_OK && o->min_speech_duration_ms >= 0) s = speech_request_set_int(r, SPEECH_OPT_MIN_SPEECH_DURATION_MS, o->min_speech_duration_ms);
    if (s == SPEECH_OK && o->min_silence_duration_ms >= 0) s = speech_request_set_int(r, SPEECH_OPT_MIN_SILENCE_DURATION_MS, o->min_silence_duration_ms);
    if (s == SPEECH_OK && o->speech_pad_ms >= 0) s = speech_request_set_int(r, SPEECH_OPT_SPEECH_PAD_MS, o->speech_pad_ms);
    if (s == SPEECH_OK && !isnan(o->max_speech_duration_s)) s = speech_request_set_float(r, SPEECH_OPT_MAX_SPEECH_DURATION_S, o->max_speech_duration_s);
    return s;
}

/** The value of an integer option in a set, or the model's default. */
static int64_t value_of(const speech_model_info * info, speech_option option, int64_t value) {
    int64_t fallback = 0;
    if (value >= 0) return value;
    speech_model_info_option_default_int(info, option, &fallback);
    return fallback;
}

/** The regions in seconds. */
typedef struct {
    int count;
    double start[MAX_REGIONS], end[MAX_REGIONS];
} Regions;

/** speech_detect()'s regions of `n` samples at `rate` with the options of `set`; 0 on a failure. */
static int whole(speech_model * model, const float * x, size_t n, int rate, const Options * set, Regions * out) {
    speech_request * r = NULL;
    const int ok = speech_request_new(model, &r) == SPEECH_OK && speech_request_set_audio(r, x, n, rate) == SPEECH_OK &&
                   set_options(r, set) == SPEECH_OK && speech_detect(r) == SPEECH_OK;
    const speech_result * result = ok ? speech_request_result(r) : NULL;
    out->count = result ? (int) speech_result_segment_count(result) : 0;
    for (int i = 0; result && i < out->count && i < MAX_REGIONS; i++) speech_result_segment(result, (size_t) i, &out->start[i], &out->end[i], NULL);
    if (!ok) fail("speech_detect");
    speech_request_free(r);
    return ok && out->count <= MAX_REGIONS;
}

/**
 * What a detection gave while its audio arrived: its regions, the samples pushed when each was given, n + 1 for those
 * that the end gave, and each start it said someone speaks from, with the samples pushed when it first said it.
 */
typedef struct {
    Regions regions;
    size_t given_at[MAX_REGIONS];
    int n_said;
    double said[MAX_SAID];
    size_t said_at[MAX_SAID];
} Streamed;

/** Reads what `d` gives after `at` samples have been pushed; 0 when it gives more regions than the check holds. */
static int record(const speech_detection * d, Streamed * s, size_t at) {
    const size_t count = speech_detection_region_count(d);
    if (count > MAX_REGIONS || count < (size_t) s->regions.count) return 0;
    for (size_t i = (size_t) s->regions.count; i < count; i++) {
        if (speech_detection_region(d, i, &s->regions.start[i], &s->regions.end[i]) != SPEECH_OK) return 0;
        s->given_at[i] = at;
    }
    s->regions.count = (int) count;
    double start = 0;
    if (speech_detection_speaking(d, &start) && (s->n_said == 0 || s->said[s->n_said - 1] != start) && s->n_said < MAX_SAID) {
        s->said[s->n_said] = start;
        s->said_at[s->n_said++] = at;
    }
    return 1;
}

/** A detection of `set` started on `model` for audio at `rate`, or NULL on a failure. */
static speech_detection * start(speech_model * model, int rate, const Options * set) {
    speech_request * r = NULL;
    speech_detection * d = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || set_options(r, set) != SPEECH_OK || speech_detection_start(r, rate, &d) != SPEECH_OK) {
        fail("speech_detection_start");
    }
    speech_request_free(r);
    return d;
}

/** Pushes `n` samples at `rate` in pieces of `kind` and ends the detection; 0 on a failure or a region that changed. */
static int stream(speech_model * model, const float * x, size_t n, int rate, const Options * set, int kind, Streamed * s) {
    memset(s, 0, sizeof *s);
    speech_detection * d = start(model, rate, set);
    if (!d) return 0;
    uint32_t seed = 20261008u;
    int ok = 1;
    for (size_t at = 0; ok && at < n;) {
        size_t k = piece(kind, rate, &seed);
        if (k > n - at) k = n - at;
        ok = speech_detection_push(d, x + at, k) == SPEECH_OK || fail("speech_detection_push");
        at += k;
        ok = ok && record(d, s, at);
    }
    ok = ok && (speech_detection_end(d) == SPEECH_OK || fail("speech_detection_end")) && record(d, s, n + 1);
    // A region given before the end has the bounds it was given with.
    for (int i = 0; ok && i < s->regions.count; i++) {
        double a = 0, b = 0;
        ok = speech_detection_region(d, (size_t) i, &a, &b) == SPEECH_OK && a == s->regions.start[i] && b == s->regions.end[i];
        if (!ok) fprintf(stderr, "FAIL: region %d changed after it was given\n", i);
    }
    ok = ok && !speech_detection_speaking(d, NULL);
    speech_detection_free(d);
    return ok;
}

static int same_regions(const Regions * a, const Regions * b) {
    int same = a->count == b->count;
    for (int i = 0; same && i < a->count; i++) same = a->start[i] == b->start[i] && a->end[i] == b->end[i];
    return same;
}

static void print_regions(const char * what, const Regions * r) {
    fprintf(stderr, "  %s:", what);
    for (int i = 0; i < r->count; i++) fprintf(stderr, " %.6f-%.6f", r->start[i], r->end[i]);
    fprintf(stderr, "%s\n", r->count ? "" : " none");
}

/**
 * Whether what a detection of `n` samples said agrees with its regions: with min_speech_duration_ms 0 each start said is
 * a region's, and where `said_first` asks, each region given before the end had its start said before it was given.
 */
static int said_well(const Streamed * s, size_t n, int min_speech_zero, int said_first) {
    for (int j = 0; min_speech_zero && j < s->n_said; j++) {
        int found = 0;
        for (int i = 0; !found && i < s->regions.count; i++) found = s->said[j] == s->regions.start[i];
        if (!found) {
            fprintf(stderr, "FAIL: someone was said to speak from %.6f s, where no region starts\n", s->said[j]);
            return 0;
        }
    }
    for (int i = 0; said_first && i < s->regions.count; i++) {
        int found = 0;
        for (int j = 0; !found && j < s->n_said; j++) found = s->said[j] == s->regions.start[i] && s->said_at[j] < s->given_at[i];
        if (!found && s->given_at[i] <= n) {
            fprintf(stderr, "FAIL: the region from %.6f s was given before anyone was said to speak from its start\n", s->regions.start[i]);
            return 0;
        }
    }
    return 1;
}

/** The audio of `n` samples at `rate` made at `to` Hz by linear interpolation; `*m` is its length. */
static float * interpolated(const float * x, size_t n, int rate, int to, size_t * m) {
    *m = (size_t) ((double) n * to / rate);
    float * y = (float *) malloc(*m * sizeof(float));
    for (size_t j = 0; j < *m; j++) {
        const double t = (double) j * rate / to;
        const size_t i = (size_t) t;
        const double f = t - (double) i;
        y[j] = (float) ((1 - f) * x[i] + f * (i + 1 < n ? x[i + 1] : 0.0f));
    }
    return y;
}

/** What a detection refuses, that its options are copied, and that the request it was started from runs as it was. */
static int check_refusals(speech_model * model, const float * x, size_t n, int rate) {
    speech_request * r = NULL;
    speech_detection * d = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_detection_start(NULL, rate, &d), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a detection of no request") && d == NULL;
    ok &= expect(speech_detection_start(r, rate, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a detection given nowhere");
    ok &= expect(speech_detection_start(r, 0, &d), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a detection at 0 Hz") && d == NULL;
    ok &= expect(speech_detection_start(r, 44101, &d), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a detection at 44101 Hz") && d == NULL;
    ok &= expect(speech_request_set_audio(r, x, n, rate), SPEECH_OK, NULL, "the request's audio") &&
          expect(speech_detection_start(r, rate, &d), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a detection of a request with audio") && d == NULL;
    speech_request_free(r);

    Regions want, got;
    if (!whole(model, x, n, rate, &SETS[0], &want) || speech_request_new(model, &r) != SPEECH_OK) return fail("a request");
    ok &= expect(speech_detection_start(r, rate, &d), SPEECH_OK, NULL, "a detection");
    // The options were copied: a threshold set on the request afterwards changes nothing.
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_THRESHOLD, 0.99), SPEECH_OK, NULL, "a threshold set after the start");
    ok &= expect(speech_detection_push(d, NULL, 5), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "NULL samples");
    ok &= expect(speech_detection_push(d, NULL, 0), SPEECH_OK, NULL, "no samples");
    ok &= expect(speech_detection_push(d, x, n), SPEECH_OK, NULL, "the audio") &&
          expect(speech_detection_end(d), SPEECH_OK, NULL, "the end");
    got.count = (int) speech_detection_region_count(d);
    for (int i = 0; i < got.count && i < MAX_REGIONS; i++) speech_detection_region(d, (size_t) i, &got.start[i], &got.end[i]);
    if (!same_regions(&want, &got)) {
        fprintf(stderr, "FAIL: a detection whose request changed after its start gave other regions\n");
        ok = 0;
    }
    ok &= expect(speech_detection_region(d, (size_t) got.count, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a region past the last");
    ok &= expect(speech_detection_end(d), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a second end");
    ok &= expect(speech_detection_push(d, x, n), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "audio after the end");
    speech_detection_free(d);
    // The request runs as it was, with the threshold it was given last.
    ok &= expect(speech_request_set_audio(r, x, n, rate), SPEECH_OK, NULL, "audio for the request") &&
          expect(speech_detect(r), SPEECH_OK, NULL, "the request a detection was started from");
    speech_request_free(r);
    speech_detection_free(NULL);
    ok &= speech_detection_region_count(NULL) == 0 && speech_detection_speaking(NULL, NULL) == 0;
    return ok ? 0 : 1;
}

/** Two detections of other options on one model, pushed in turn in pieces of 20 ms, each against speech_detect(). */
static int check_two(speech_model * model, const float * x, size_t n, int rate) {
    const Options * sets[2] = {&SETS[0], &SETS[2]};
    speech_detection * d[2] = {start(model, rate, sets[0]), start(model, rate, sets[1])};
    int ok = d[0] && d[1];
    for (size_t at = 0; ok && at < n; at += (size_t) rate / 50) {
        const size_t k = n - at < (size_t) rate / 50 ? n - at : (size_t) rate / 50;
        ok = speech_detection_push(d[0], x + at, k) == SPEECH_OK && speech_detection_push(d[1], x + at, k) == SPEECH_OK;
    }
    for (int i = 0; i < 2; i++) {
        Regions want, got;
        ok = ok && speech_detection_end(d[i]) == SPEECH_OK && whole(model, x, n, rate, sets[i], &want);
        got.count = ok ? (int) speech_detection_region_count(d[i]) : 0;
        for (int k = 0; k < got.count && k < MAX_REGIONS; k++) speech_detection_region(d[i], (size_t) k, &got.start[k], &got.end[k]);
        ok = ok && same_regions(&want, &got);
        speech_detection_free(d[i]);
    }
    printf("two detections on one model pushed in turn: %s\n", ok ? "each speech_detect()'s regions" : "DIFFERENT");
    return ok ? 0 : 1;
}

/** Times `n` samples at `rate` pushed in pieces of 20 ms, 100 ms and 1 s with the default options. */
static int time_pieces(speech_model * model, const float * x, size_t n, int rate) {
    for (int kind = 1; kind <= 3; kind++) {
        speech_detection * d = start(model, rate, &SETS[0]);
        if (!d) return 1;
        uint32_t seed = 0;
        const size_t size = piece(kind, rate, &seed);
        double longest = 0;
        const double t0 = now_seconds();
        for (size_t at = 0; at < n; at += size) {
            const double p0 = now_seconds();
            if (speech_detection_push(d, x + at, n - at < size ? n - at : size) != SPEECH_OK) return fail("speech_detection_push");
            const double took = now_seconds() - p0;
            if (took > longest) longest = took;
        }
        if (speech_detection_end(d) != SPEECH_OK) return fail("speech_detection_end");
        printf("%.2f s of audio at %d Hz in pieces of %s: %.3f s, the longest push %.2f ms\n", (double) n / rate, rate, PIECES[kind],
               now_seconds() - t0, longest * 1000);
        speech_detection_free(d);
    }
    return 0;
}

int check_detection_stream(speech_model * model, float * const * audio, const size_t * lengths, const char * const * names, int n_dumps) {
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info);
    const int rates[3] = {rate, 24000, 48000};
    int ok = 1, longest = 0;
    static Streamed got;
    for (int d = 0; d < n_dumps; d++) {
        if (lengths[d] > lengths[longest]) longest = d;
        for (int k = 0; k < 3; k++) {
            size_t n = lengths[d];
            float * x = k == 0 ? audio[d] : interpolated(audio[d], lengths[d], rate, rates[k], &n);
            const double t0 = now_seconds();
            int streams = 0, regions = 0, equal = 1;
            for (int s = 0; s < N_SETS; s++) {
                const Options * set = &SETS[s];
                const int64_t min_speech = value_of(info, SPEECH_OPT_MIN_SPEECH_DURATION_MS, set->min_speech_duration_ms);
                const int64_t min_silence = value_of(info, SPEECH_OPT_MIN_SILENCE_DURATION_MS, set->min_silence_duration_ms);
                const int64_t pad = value_of(info, SPEECH_OPT_SPEECH_PAD_MS, set->speech_pad_ms);
                Regions want;
                if (!whole(model, x, n, rates[k], set, &want)) return 1;
                regions += want.count;
                // Every set with random sizes; the sets of dump.py's defaults, a region's end that waits and a limit with
                // the pieces of every size.
                for (int kind = 0; kind < N_PIECES; kind++) {
                    if (kind != N_PIECES - 1 && s != 0 && s != 3 && s != 5) continue;
                    if (!stream(model, x, n, rates[k], set, kind, &got)) return 1;
                    streams++;
                    const int same = same_regions(&want, &got.regions);
                    const int said = said_well(&got, n, min_speech == 0, kind == 0 && min_silence >= 2 * pad && isnan(set->max_speech_duration_s));
                    if (!same || !said) {
                        fprintf(stderr, "FAIL: %s at %d Hz, %s, pieces of %s: %s\n", names[d], rates[k], set->name, PIECES[kind],
                                same ? "what it said disagrees with its regions" : "the regions differ from speech_detect()'s");
                        print_regions("speech_detect()", &want);
                        print_regions("pushed", &got.regions);
                        equal = 0;
                    }
                }
            }
            printf("%s at %d Hz: %d detections of %d sets pushed in pieces, %d regions in all, %s, in %.2f s\n", names[d], rates[k], streams, N_SETS,
                   regions, equal ? "speech_detect()'s" : "DIFFERENT", now_seconds() - t0);
            ok = ok && equal;
            if (k != 0) free(x);
        }
    }
    speech_model_info_free(info);
    if (!ok) return 1;
    if (check_two(model, audio[longest], lengths[longest], rate) != 0 || time_pieces(model, audio[longest], lengths[longest], rate) != 0) return 1;
    return check_refusals(model, audio[longest], lengths[longest], rate);
}
