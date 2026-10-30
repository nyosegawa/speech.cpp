#include "boosting.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <stdexcept>
#include <tuple>

namespace fastconformer {

namespace {

/** A node of ContextGraph: its children in the order they were added, as a Python dict keeps them. */
struct Node {
    int32_t token;
    double token_score, node_score;
    bool is_end;
    int level;
    int fail = 0;
    std::vector<std::pair<int32_t, int>> children;
    std::map<int32_t, int> child_of;

    int child(int32_t t) const {
        const auto it = child_of.find(t);
        return it == child_of.end() ? -1 : it->second;
    }
};

/** ContextGraph.build() with uniform weights off and every word at the graph's context score, then its fail arcs. */
std::vector<Node> context_graph(const std::vector<std::vector<int32_t>> & words, double context_score, double depth_scaling) {
    std::vector<Node> nodes = {{-1, 0.0, 0.0, false, 0}};
    for (const std::vector<int32_t> & tokens : words) {
        int node = 0;
        for (size_t i = 0; i < tokens.size(); i++) {
            const bool last = i + 1 == tokens.size();
            int child = nodes[(size_t) node].child(tokens[i]);
            if (child < 0) {
                // The depth scaling gives every token after the first a larger bonus.
                const double token_score = i > 0 ? context_score * depth_scaling + std::log((double) (i + 1)) : context_score;
                child = (int) nodes.size();
                nodes.push_back({tokens[i], token_score, nodes[(size_t) node].node_score + token_score, last, (int) i + 1});
                nodes[(size_t) node].children.push_back({tokens[i], child});
                nodes[(size_t) node].child_of.emplace(tokens[i], child);
            } else {
                Node & c = nodes[(size_t) child];
                c.token_score = std::max(context_score, c.token_score);
                c.node_score = nodes[(size_t) node].node_score + c.token_score;
                c.is_end = last || c.is_end;
            }
            node = child;
        }
    }
    // _fill_fail_output(): breadth first from the root, the fail arc of a node is the longest proper suffix of its
    // tokens that is a node.
    std::deque<int> queue;
    for (const auto & [token, child] : nodes[0].children) queue.push_back(child);
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop_front();
        for (const auto & [token, child] : nodes[(size_t) current].children) {
            int fail = nodes[(size_t) current].fail;
            if (nodes[(size_t) fail].child(token) >= 0) {
                fail = nodes[(size_t) fail].child(token);
            } else {
                fail = nodes[(size_t) fail].fail;
                while (nodes[(size_t) fail].child(token) < 0) {
                    fail = nodes[(size_t) fail].fail;
                    if (nodes[(size_t) fail].token == -1) break;
                }
                if (nodes[(size_t) fail].child(token) >= 0) fail = nodes[(size_t) fail].child(token);
            }
            nodes[(size_t) child].fail = fail;
            queue.push_back(child);
        }
    }
    return nodes;
}

}  // namespace

BoostingTree::BoostingTree(const std::vector<std::vector<int32_t>> & words, int32_t vocab_size, double context_score, double depth_scaling,
                           float unk_score)
    : vocab_size_(vocab_size) {
    for (const std::vector<int32_t> & tokens : words) {
        for (int32_t t : tokens) {
            if (t < 0 || t >= vocab_size) throw std::logic_error("a boosted word holds a token outside the vocabulary");
        }
    }
    const std::vector<Node> nodes = context_graph(words, context_score, depth_scaling);
    // _read_context_graph(): the branches breadth first, which come one order after another.
    struct Branch {
        int32_t token;
        int start, next;
    };
    std::vector<std::vector<Branch>> orders;
    std::deque<int> queue = {0};
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop_front();
        for (const auto & [token, child] : nodes[(size_t) current].children) {
            const size_t order = (size_t) nodes[(size_t) child].level;
            if (orders.size() < order) orders.resize(order);
            orders[order - 1].push_back({token, current, child});
            queue.push_back(child);
        }
    }
    // BoostingTreeStorage: the start state's arc of each token is that token's arc, a word's first token leading to
    // its state and any other token back to the start; each order's branches are added sorted by the node they leave
    // and their token. A state that ends a word does not back off with a penalty.
    std::map<int, int32_t> state_of = {{0, kStart}};
    states_.push_back({0, vocab_size, 1, kStart, 0.0f});
    arcs_.resize((size_t) vocab_size);
    for (int32_t t = 0; t < vocab_size; t++) arcs_[(size_t) t] = {kStart, kStart, t, unk_score};
    for (size_t o = 0; o < orders.size(); o++) {
        std::vector<Branch> branches = orders[o];
        std::sort(branches.begin(), branches.end(), [](const Branch & a, const Branch & b) { return std::tie(a.start, a.token) < std::tie(b.start, b.token); });
        for (const Branch & b : branches) {
            const Node & next = nodes[(size_t) b.next];
            const int32_t from = state_of.at(b.start), state = (int32_t) states_.size();
            const float backoff = next.is_end ? 0.0f : (float) (nodes[(size_t) next.fail].node_score - next.node_score);
            states_.push_back({0, 0, states_[(size_t) from].order + 1, state_of.at(next.fail), backoff});
            state_of[b.next] = state;
            if (o == 0) {
                arcs_[(size_t) b.token] = {kStart, state, b.token, (float) next.token_score};
                continue;
            }
            const int32_t arc = (int32_t) arcs_.size();
            arcs_.push_back({from, state, b.token, (float) next.token_score});
            State & s = states_[(size_t) from];
            if (s.arcs_start == 0) s.arcs_start = arc;
            s.arcs_end = arc + 1;
        }
    }
}

void BoostingTree::advance(int32_t state, std::vector<float> & scores, std::vector<int32_t> & next) const {
    scores.assign((size_t) vocab_size_, 0.0f);
    next.assign((size_t) vocab_size_, -1);
    float backoff = 0.0f;
    for (int32_t s = state;;) {
        const State & st = states_[(size_t) s];
        for (int32_t a = st.arcs_start; a < st.arcs_end; a++) {
            const Arc & arc = arcs_[(size_t) a];
            if (next[(size_t) arc.token] >= 0) continue;
            scores[(size_t) arc.token] = backoff + arc.weight;
            next[(size_t) arc.token] = arc.to;
        }
        if (s == kStart) break;
        backoff += st.backoff_weight;
        s = st.backoff_to;
    }
}

}  // namespace fastconformer
