#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * SHA-256 (FIPS 180-4) over bytes given in pieces, for checking a fetched model file against the catalog without a
 * library: speech links nothing but ggml and cpp-httplib, and cpp-httplib computes hashes only through OpenSSL.
 */
class Sha256 {
public:
    Sha256();
    void update(const void * data, size_t size);
    /** The digest of every byte given, as 64 lower-case hexadecimal digits; the object takes no more bytes after it. */
    std::string hex();

private:
    void block(const uint8_t * p);

    uint32_t state_[8];
    uint8_t buffer_[64];
    size_t buffered_ = 0;
    uint64_t length_ = 0;
};
