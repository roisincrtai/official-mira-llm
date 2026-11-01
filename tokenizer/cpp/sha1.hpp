// tokenizer/cpp/sha1.hpp -- SHA-1 (FIPS 180-4), for the corpus signature; hexdigest as Python's
// hashlib.sha1().hexdigest().
//
// Standard C++17 only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mira_bpe {

class Sha1 {
public:
    void update(std::string_view data) {
        for (unsigned char c : data) {
            block_[used_++] = c;
            length_ += 8;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    std::string hexdigest() {
        std::uint64_t bits = length_;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            while (used_ < 64) block_[used_++] = 0;
            compress();
            used_ = 0;
        }
        while (used_ < 56) block_[used_++] = 0;
        for (int k = 7; k >= 0; --k) block_[used_++] = static_cast<unsigned char>(bits >> (8 * k));
        compress();
        static const char* HEX = "0123456789abcdef";
        std::string out;
        for (std::uint32_t word : h_)
            for (int shift = 28; shift >= 0; shift -= 4) out.push_back(HEX[(word >> shift) & 0xF]);
        return out;
    }

private:
    static std::uint32_t rotate(std::uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    void compress() {
        std::uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(block_[4 * i]) << 24) | (std::uint32_t(block_[4 * i + 1]) << 16) |
                   (std::uint32_t(block_[4 * i + 2]) << 8) | std::uint32_t(block_[4 * i + 3]);
        for (int i = 16; i < 80; ++i) w[i] = rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
        for (int i = 0; i < 80; ++i) {
            std::uint32_t f, k;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            std::uint32_t t = rotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotate(b, 30);
            b = a;
            a = t;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
    }
    std::uint32_t h_[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    unsigned char block_[64] = {};
    std::size_t used_ = 0;
    std::uint64_t length_ = 0;
};

}  // namespace mira_bpe
