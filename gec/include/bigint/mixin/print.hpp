#pragma once
#ifndef GEC_BIGINT_MIXIN_PRINT_HPP
#define GEC_BIGINT_MIXIN_PRINT_HPP

#include <utils/basic.hpp>
#include <utils/crtp.hpp>

#include <limits>
#include <type_traits>

#include <cstdio>

namespace gec {

namespace print {
template <typename T>
GEC_HD void print(const T &data) {
    const char *it = reinterpret_cast<const char *>(&data);
    const char *end = it + sizeof(T);
    printf("Unknown {");
    if (it < end) {
        printf("%02x", *it++);
        for (; it < end; ++it) {
            printf(" %02x", *it);
        }
    }
    printf("}");
}

template <typename T, size_t s = sizeof(T)>
struct FormatStr;

template <>
struct FormatStr<unsigned int, 2> {
    GEC_HD GEC_INLINE static const char *call() { return "%04x"; }
};
template <>
struct FormatStr<unsigned int, 4> {
    GEC_HD GEC_INLINE static const char *call() { return "%08x"; }
};
template <>
struct FormatStr<unsigned int, 8> {
    GEC_HD GEC_INLINE static const char *call() { return "%016x"; }
};
template <>
inline GEC_HD void print<unsigned int>(const unsigned int &data) {
    printf(FormatStr<unsigned int>::call(), data);
}

template <>
struct FormatStr<unsigned long, 4> {
    GEC_HD GEC_INLINE static const char *call() { return "%08lx"; }
};
template <>
struct FormatStr<unsigned long, 8> {
    GEC_HD GEC_INLINE static const char *call() { return "%016lx"; }
};
template <>
inline GEC_HD void print<unsigned long>(const unsigned long &data) {
    printf(FormatStr<unsigned long>::call(), data);
}

template <>
struct FormatStr<unsigned long long, 4> {
    GEC_HD GEC_INLINE static const char *call() { return "%08llx"; }
};
template <>
struct FormatStr<unsigned long long, 8> {
    GEC_HD GEC_INLINE static const char *call() { return "%016llx"; }
};
template <>
inline GEC_HD void print<unsigned long long>(const unsigned long long &data) {
    printf(FormatStr<unsigned long long>::call(), data);
}

} // namespace print

namespace bigint {

/** @brief mixin that enables output array() with stdio
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES ArrayPrint
    : protected CRTP<Core, ArrayPrint<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, ArrayPrint<Core, LIMB_T, LIMB_N>>;

  public:
    GEC_HD void print() const {
        printf("0x");
        print::print(this->core().array()[LIMB_N - 1]);
        for (size_t i = 1; i < LIMB_N; ++i) {
            printf(" ");
            print::print(this->core().array()[LIMB_N - 1 - i]);
        }
    }
    GEC_HD void println() const {
        print();
        printf("\n");
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_PRINT_HPP
