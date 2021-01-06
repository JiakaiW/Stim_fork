#include "chp_sim.h"

ChpSim::ChpSim(size_t num_qubits) : inv_state(Tableau::identity(num_qubits)), rng((std::random_device {})()) {
}

bool ChpSim::is_deterministic(size_t target) const {
    size_t n = inv_state.num_qubits;
    const auto &z_at_beginning = inv_state.z_obs_ptr(target);
    for (size_t q = 0; q < n; q++) {
        if (z_at_beginning.get_x_bit(q)) {
            return false;
        }
    }
    return true;
}

bool ChpSim::measure(size_t target, float bias) {
    size_t n = inv_state.num_qubits;
    const auto &z_obs = inv_state.z_obs_ptr(target);

    // X or Y observable indicates random result.
    size_t pivot = UINT32_MAX;
    for (size_t q = 0; q < n; q++) {
        if (z_obs.get_x_bit(q)) {
            pivot = q;
            break;
        }
    }

    if (pivot == UINT32_MAX) {
        // Deterministic result.
        return *z_obs.ptr_sign;
    }

    // Cancel out other X / Y components.
    for (size_t q = pivot + 1; q < n; q++) {
        if (z_obs.get_x_bit(q)) {
            inv_state.inplace_scatter_append(
                    GATE_TABLEAUS.at("CNOT"),
                    {pivot, q});
        }
    }

    // Collapse the state.
    if (z_obs.get_z_bit(pivot)) {
        inv_state.inplace_scatter_append(
                GATE_TABLEAUS.at("H_YZ"),
                {pivot});
    } else {
        inv_state.inplace_scatter_append(
                GATE_TABLEAUS.at("H_XZ"),
                {pivot});
    }

    auto coin_flip = std::bernoulli_distribution(bias)(rng);
    if (*z_obs.ptr_sign != coin_flip) {
        inv_state.inplace_scatter_append(
                GATE_TABLEAUS.at("X"),
                {pivot});
    }

    return coin_flip;
}

void ChpSim::H(size_t q) {
    inv_state.inplace_scatter_prepend_H(q);
}

void ChpSim::S(size_t q) {
    inv_state.inplace_scatter_prepend(GATE_TABLEAUS.at("S_DAG"), {q});
}

void ChpSim::CNOT(size_t c, size_t t) {
    inv_state.inplace_scatter_prepend_CNOT(c, t);
}

void ChpSim::CZ(size_t c, size_t t) {
    inv_state.inplace_scatter_prepend_CZ(c, t);
}

void ChpSim::X(size_t q) {
    inv_state.inplace_scatter_prepend_X(q);
}

void ChpSim::Y(size_t q) {
    inv_state.inplace_scatter_prepend_Y(q);
}

void ChpSim::Z(size_t q) {
    inv_state.inplace_scatter_prepend_Z(q);
}