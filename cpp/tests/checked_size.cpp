#include "qfin/checked_size.hpp"
#include <limits>
#include <stdexcept>

int main() {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (qfin::checked_multiply(maximum, 0) != 0 ||
        qfin::checked_multiply(maximum, 1) != maximum ||
        qfin::checked_add(maximum, 0) != maximum ||
        qfin::checked_allocation(0, maximum) != 0) { return 1; }
    int rejected = 0;
    try { qfin::checked_multiply(maximum / 2 + 1, 2); }
    catch (const std::length_error&) { ++rejected; }
    try { qfin::checked_add(maximum, 1); }
    catch (const std::length_error&) { ++rejected; }
    try { qfin::checked_allocation(qfin::max_operation_bytes / 8 + 1); }
    catch (const std::length_error&) { ++rejected; }
    try { qfin::checked_allocation(maximum, maximum); }
    catch (const std::length_error&) { ++rejected; }
    return rejected == 4 ? 0 : 1;
}
