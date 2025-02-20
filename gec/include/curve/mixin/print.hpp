#pragma once
#ifndef GEC_CURVE_MIXIN_PRINT_HPP
#define GEC_CURVE_MIXIN_PRINT_HPP

#include <cstdio>
#include <utils/crtp.hpp>

namespace gec {

namespace curve {

template <typename Point, size_t I, size_t N>
struct GEC_EMPTY_BASES PointPrintHelper {
    GEC_HD GEC_INLINE static void call(const Point &point) {
        printf(",\n ");
        point.template get<I>().print();
        PointPrintHelper<Point, I + 1, N>::call(point);
    }
};
template <typename Point, size_t N>
struct GEC_EMPTY_BASES PointPrintHelper<Point, 0, N> {
    GEC_HD GEC_INLINE static void call(const Point &point) {
        point.template get<0>().print();
        PointPrintHelper<Point, 1, N>::call(point);
    }
};
template <typename Point, size_t N>
struct GEC_EMPTY_BASES PointPrintHelper<Point, N, N> {
    GEC_HD GEC_INLINE static void call(const Point &) {}
};
template <typename Point>
struct GEC_EMPTY_BASES PointPrintHelper<Point, 0, 0> {
    GEC_HD GEC_INLINE static void call(const Point &) {}
};

/** @brief mixin that enables output x() and y() with stdio
 */
template <typename Core>
class GEC_EMPTY_BASES PointPrint : protected CRTP<Core, PointPrint<Core>> {
    friend CRTP<Core, PointPrint<Core>>;

  public:
    GEC_HD void print() const {
        using namespace std;
        printf("{");
        PointPrintHelper<Core, 0, Core::CompN>::call(this->core());
        printf("}\n");
    }
    GEC_HD void println() const {
        this->print();
        printf("\n");
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_PRINT_HPP
