#pragma once
#ifndef GEC_BIGINT_MIXIN_VT_COMPARE_HPP
#define GEC_BIGINT_MIXIN_VT_COMPARE_HPP

#include <utils/crtp.hpp>
#include <utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief TODO:
 */
template <class Core, class LIMB_T, size_t LIMB_N>
struct GEC_EMPTY_BASES VtCompare
    : protected CRTP<Core, VtCompare<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, VtCompare<Core, LIMB_T, LIMB_N>>;

    GEC_HD GEC_INLINE utils::CmpEnum cmp(const Core &other) const {
        return utils::VtSeqCmp<LIMB_N, LIMB_T>::call(this->core().array(),
                                                     other.array());
    }
    GEC_HD GEC_INLINE bool operator==(const Core &other) const {
        return utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            this->core().array(), other.array());
    }
    GEC_HD GEC_INLINE bool operator!=(const Core &other) const {
        return !(this->core() == other);
    }
    GEC_HD GEC_INLINE bool operator<(const Core &other) const {
        return this->cmp(other) == utils::CmpEnum::Lt;
    }
    GEC_HD GEC_INLINE bool operator>=(const Core &other) const {
        return !(this->core() < other);
    }
    GEC_HD GEC_INLINE bool operator>(const Core &other) const {
        return this->cmp(other) == utils::CmpEnum::Gt;
    }
    GEC_HD GEC_INLINE bool operator<=(const Core &other) const {
        return !(this->core() > other);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_VT_COMPARE_HPP
