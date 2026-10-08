/*
 * What speech-api-check needs of the operating system, in the way of each: opening a file by a UTF-8 path, the time,
 * threads, and a mutex with a condition variable.
 */

#include <stdlib.h>

#include "speech-api-check.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

#ifdef _WIN32
FILE * open_utf8(const char * path, const char * mode) {
    wchar_t wide[4096], wide_mode[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, 4096) || !MultiByteToWideChar(CP_UTF8, 0, mode, -1, wide_mode, 8)) return NULL;
    return _wfopen(wide, wide_mode);
}
#else
FILE * open_utf8(const char * path, const char * mode) {
    return fopen(path, mode);
}
#endif

double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER count, frequency;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&frequency);
    return (double) count.QuadPart / (double) frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
#endif
}

void sleep_seconds(double seconds) {
#ifdef _WIN32
    Sleep((DWORD) (seconds * 1000));
#else
    struct timespec t;
    t.tv_sec = (time_t) seconds;
    t.tv_nsec = (long) ((seconds - (double) t.tv_sec) * 1e9);
    nanosleep(&t, NULL);
#endif
}

typedef struct {
    void (*body)(void * arg);
    void * arg;
} ThreadStart;

#ifdef _WIN32
static DWORD WINAPI thread_main(LPVOID user_data) {
#else
static void * thread_main(void * user_data) {
#endif
    ThreadStart * s = (ThreadStart *) user_data;
    s->body(s->arg);
    free(s);
    return 0;
}

void thread_start(Thread * thread, void (*body)(void * arg), void * arg) {
    ThreadStart * s = (ThreadStart *) malloc(sizeof(ThreadStart));
    s->body = body;
    s->arg = arg;
#ifdef _WIN32
    thread->handle = CreateThread(NULL, 0, thread_main, s, 0, NULL);
#else
    pthread_t * t = (pthread_t *) malloc(sizeof(pthread_t));
    pthread_create(t, NULL, thread_main, s);
    thread->handle = t;
#endif
}

void thread_join(Thread * thread) {
#ifdef _WIN32
    WaitForSingleObject((HANDLE) thread->handle, INFINITE);
    CloseHandle((HANDLE) thread->handle);
#else
    pthread_join(*(pthread_t *) thread->handle, NULL);
    free(thread->handle);
#endif
}

void monitor_init(Monitor * m) {
#ifdef _WIN32
    m->lock = malloc(sizeof(CRITICAL_SECTION));
    m->changed = malloc(sizeof(CONDITION_VARIABLE));
    InitializeCriticalSection((CRITICAL_SECTION *) m->lock);
    InitializeConditionVariable((CONDITION_VARIABLE *) m->changed);
#else
    m->lock = malloc(sizeof(pthread_mutex_t));
    m->changed = malloc(sizeof(pthread_cond_t));
    pthread_mutex_init((pthread_mutex_t *) m->lock, NULL);
    pthread_cond_init((pthread_cond_t *) m->changed, NULL);
#endif
}

void monitor_free(Monitor * m) {
#ifdef _WIN32
    DeleteCriticalSection((CRITICAL_SECTION *) m->lock);
#else
    pthread_mutex_destroy((pthread_mutex_t *) m->lock);
    pthread_cond_destroy((pthread_cond_t *) m->changed);
#endif
    free(m->lock);
    free(m->changed);
}

void monitor_lock(Monitor * m) {
#ifdef _WIN32
    EnterCriticalSection((CRITICAL_SECTION *) m->lock);
#else
    pthread_mutex_lock((pthread_mutex_t *) m->lock);
#endif
}

void monitor_unlock(Monitor * m) {
#ifdef _WIN32
    LeaveCriticalSection((CRITICAL_SECTION *) m->lock);
#else
    pthread_mutex_unlock((pthread_mutex_t *) m->lock);
#endif
}

void monitor_wait(Monitor * m) {
#ifdef _WIN32
    SleepConditionVariableCS((CONDITION_VARIABLE *) m->changed, (CRITICAL_SECTION *) m->lock, INFINITE);
#else
    pthread_cond_wait((pthread_cond_t *) m->changed, (pthread_mutex_t *) m->lock);
#endif
}

void monitor_signal(Monitor * m) {
#ifdef _WIN32
    WakeAllConditionVariable((CONDITION_VARIABLE *) m->changed);
#else
    pthread_cond_broadcast((pthread_cond_t *) m->changed);
#endif
}
