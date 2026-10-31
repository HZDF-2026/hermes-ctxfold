// core.h — ctxfold deterministic core (C++17, zero dependencies).
//
// Ported from the JS reference engine (ctxfold v2.0.0, H:/ctxfold/ctxfold.js)
// and from the earlier C++ port used by babele (H:/work/greenroom/src/ctxfold.cpp).
// Semantic contract: same input -> same output as the JS engine for every
// primitive here (anchors / tok / fold / gist / distill / entryId / recall).
// Byte-determinism is what makes the memory layer auditable and lets every
// host (dsh, Pi, Torre di Babele, Hermes) share one memory library.
#ifndef HERMES_CTXFOLD_CORE_H
#define HERMES_CTXFOLD_CORE_H

#include <cstddef>
#include <string>
#include <vector>

namespace cf {

// Approximate token count (CJK ~= 1/code point, Latin ~= 4 chars/token,
// space and newline free). Mirrors the JS engine's tok().
long long tok(const std::string& text);

// Collapse whitespace runs to one space and trim (JS normText).
std::string normText(const std::string& text);

// Verbatim anchor layer: Windows/POSIX paths, URLs, sha1/sha256, numbers+units,
// command starts, dotted identifiers, backtick code spans. First-seen order,
// capped at `limit`. Never paraphrased.
std::vector<std::string> extractAnchors(const std::string& text,
                                        std::size_t limit = 200);

// Keep the head and tail lines, replace the middle with a verifiable marker.
std::string fold(const std::string& text, std::size_t keepHead,
                 std::size_t keepTail);

// First `maxLen` code points of the whitespace-normalized text, plus a
// truncation marker carrying the original token count.
std::string gist(const std::string& text, std::size_t maxLen);

struct Distilled {
    std::vector<std::string> facts;      // anchors (verbatim facts)
    std::vector<std::string> lessons;    // rule-matched lesson sentences
    std::vector<std::string> decisions;  // rule-matched decision sentences
};

// Deterministic distillation. UTF-8 in/out. Same input -> same output.
Distilled distillText(const std::string& text, std::size_t maxFacts = 40,
                      std::size_t maxLessons = 12,
                      std::size_t maxDecisions = 12);

// sha256(topic + "\n" + normalized(text)), first 12 hex chars.
// The same fact re-distilled yields the same id (upsert = dedupe).
std::string entryId(const std::string& topic, const std::string& text);

// Relevance of a stored entry to a query, for recall ranking:
// stored-anchor match x3 + token overlap scaled by entry size
// + 0.15 per prior hit + 0.3 for lessons.
double recallScore(const std::string& query, const std::string& entryText,
                   const std::string& entryTopic,
                   const std::vector<std::string>& entryAnchors, int hits,
                   const std::string& kind);

}  // namespace cf

#endif  // HERMES_CTXFOLD_CORE_H
