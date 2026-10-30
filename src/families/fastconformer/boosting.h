#pragma once

#include <cstdint>
#include <vector>

namespace fastconformer {

/**
 * NeMo's phrase-boosting tree as Haas-ASR compiles it (haas_asr.boosting._compiled()): a ContextGraph of the words'
 * token sequences (nemo/collections/asr/parts/context_biasing/context_graph_universal.py), a trie with Aho-Corasick's
 * fail arcs, read into a GPUBoostingTreeModel (boosting_graph_batched.py), an acceptor over the tokens whose states back
 * off to the state of their fail node, and whose start state has an arc of every token.
 *
 * A token's bonus deep in a word is context_score * depth_scaling + log(depth), and its first context_score; a state
 * that ends no word backs off with the bonus of the word so far taken back, as its fail node's accumulated bonus less
 * its own. The bonuses are summed in double precision as Python sums them and kept in float, as NeMo's tables keep them.
 */
class BoostingTree {
public:
    /** An arc: the state it leaves, the state it reaches, its token and its bonus. */
    struct Arc {
        int32_t from, to, token;
        float weight;
    };

    /** A state: its arcs [arcs_start, arcs_end), its order, the state it backs off to and the bonus of backing off. */
    struct State {
        int32_t arcs_start, arcs_end, order, backoff_to;
        float backoff_weight;
    };

    /**
     * The tree of `words`, each a token sequence, in their order, over a vocabulary of `vocab_size` tokens (the
     * tokenizer's pieces, without the blank). An empty sequence adds nothing.
     */
    BoostingTree(const std::vector<std::vector<int32_t>> & words, int32_t vocab_size, double context_score, double depth_scaling, float unk_score);

    /** The state a decoding starts in. */
    static constexpr int32_t kStart = 0;

    /**
     * The bonus of each token from `state` [vocab_size] and the state it reaches, as GPUBoostingTreeModel.advance()
     * gives them: a token is scored by the first state on the way back to the start whose arcs hold it, with the bonuses
     * of backing off to that state summed in float.
     */
    void advance(int32_t state, std::vector<float> & scores, std::vector<int32_t> & next) const;

    const std::vector<Arc> & arcs() const { return arcs_; }
    const std::vector<State> & states() const { return states_; }
    int32_t vocab_size() const { return vocab_size_; }

private:
    int32_t vocab_size_;
    std::vector<Arc> arcs_;
    std::vector<State> states_;
};

}  // namespace fastconformer
