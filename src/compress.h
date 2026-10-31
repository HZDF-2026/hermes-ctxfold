// compress.h — deterministic message-list compaction plan (zero LLM).
//
// This is the policy layer on top of core.h: it decides, for every message,
// whether to keep it verbatim, fold it (tool output -> head/tail + hash
// marker), gist it (first N chars + truncation marker), or tombstone it
// (empty content). It never *deletes* a message, so tool_call/tool_result
// pairing always stays intact — Hermes never has to repair the list.
//
// Differences from the JS engine (documented, intentional):
//  - no template mining (the JS `templatize` collapse turned every surviving
//    tool line into a bare `⟨§id⟩`, destroying readable output);
//  - tombstone text is supplied by the host, not hardcoded here.
//
// Budget semantics: `budget` is the token ceiling for the WHOLE list after
// compaction, including the anchor block the host prepends.
#ifndef HERMES_CTXFOLD_COMPRESS_H
#define HERMES_CTXFOLD_COMPRESS_H

#include <string>
#include <vector>

namespace cf {

struct CompressOptions {
    long long budget = 60000;  // post-compaction token ceiling for the list
    int toolFold = 400;        // fold a tool result longer than this (tokens)
    int head = 6;              // lines kept at the head of a folded block
    int tail = 4;              // lines kept at the tail of a folded block
    int recent = 20;           // newest messages kept verbatim
    int gistLen = 120;         // gist length in code points
};

enum class Action { Keep, Rewrite, Tomb };

struct MessagePlan {
    Action action = Action::Keep;
    std::string text;  // new content for Rewrite; empty otherwise
};

struct CompressStats {
    long long inTok = 0;
    long long outTok = 0;
    int kept = 0;
    int tomb = 0;
    int gisted = 0;
    int folded = 0;
    int anchors = 0;
};

struct CompressPlan {
    std::vector<MessagePlan> messages;   // one entry per input message
    std::vector<std::string> anchors;    // verbatim fact layer
    CompressStats stats;
};

// `kinds[i]` is "system" | "tool" | "chat"; `texts[i]` is the UTF-8 content.
// Both vectors must be the same length. Deterministic: same input -> same plan.
CompressPlan planCompression(const std::vector<std::string>& kinds,
                             const std::vector<std::string>& texts,
                             const CompressOptions& opts);

}  // namespace cf

#endif  // HERMES_CTXFOLD_COMPRESS_H
