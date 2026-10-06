#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "speech.h"

/** The kind of a device as the tools write it: "cpu", "gpu" or "igpu". */
inline const char * device_kind_name(size_t index) {
    speech_device_kind kind = SPEECH_DEVICE_CPU;
    if (speech_device_get_kind(index, &kind) != SPEECH_OK) throw std::runtime_error(speech_last_error());
    return kind == SPEECH_DEVICE_GPU ? "gpu" : kind == SPEECH_DEVICE_IGPU ? "igpu" : "cpu";
}

/** Prints the devices the library can run on, one line each with its kind and memory, for a command-line tool. */
inline void list_devices(FILE * out) {
    for (size_t i = 0; i < speech_device_count(); i++) {
        uint64_t total = 0, free = 0;
        if (speech_device_memory(i, &total, &free) != SPEECH_OK) throw std::runtime_error(speech_last_error());
        std::fprintf(out, "%-10s %-5s %s, %.1f GB, %.1f GB free\n", speech_device_name(i), device_kind_name(i), speech_device_description(i),
                     total / 1e9, free / 1e9);
    }
    std::fflush(out);
}
