#pragma once

#include <cstdio>
#include <stdexcept>

#include "speech.h"

/** Prints the devices the library can run on, one line each with its kind and memory, for a command-line tool. */
inline void list_devices(FILE * out) {
    for (size_t i = 0; i < speech_device_count(); i++) {
        speech_device d;
        if (speech_device_get(i, &d) != SPEECH_OK) throw std::runtime_error(speech_last_error());
        // An accelerator (BLAS) runs with the CPU and is not a device to choose.
        if (d.kind == SPEECH_DEVICE_ACCEL) continue;
        const char * kind = d.kind == SPEECH_DEVICE_GPU ? "gpu" : d.kind == SPEECH_DEVICE_IGPU ? "igpu" : "cpu";
        std::fprintf(out, "%-10s %-5s %s, %.1f GB, %.1f GB free\n", d.name, kind, d.description, d.memory_total / 1e9,
                     d.memory_free / 1e9);
    }
    std::fflush(out);
}
