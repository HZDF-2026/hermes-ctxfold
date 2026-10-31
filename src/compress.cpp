// compress.cpp — two-pass deterministic budget allocation. See compress.h.
//
// Pass 1 walks the list once: token cost, anchor registry, and a per-message
// value = (1 + new-anchor count x 0.5). Pass 2 spends the budget with a
// histogram cut over those values: everything below `tombCut` is tombstoned,
// everything below `gistCut` is gisted, the rest is kept. System messages and
// the newest `recent` messages are never touched.
//
// Cost model: a message that will be emitted verbatim (system / protected
// tail) is charged its real token count; a message the plan may rewrite is
// charged its post-fold cost, so the budget reflects what will actually be sent.
//
// Anchor scanning is bounded: a tool result is scanned only for its first
// kAnchorScanChars characters. Tool bodies are rerunnable noise, their
// meaningful facts live in the head, and std::regex over megabytes of JSON
// would otherwise dominate the whole pass.
#include "compress.h"

#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>

#include "core.h"

namespace cf {
namespace {

const double kInf = std::numeric_limits<double>::infinity();
const std::size_t kAnchorScanChars = 1024;    // tool output: head only
const std::size_t kChatScanChars = 64 * 1024;  // chat: generous but bounded
const std::size_t kAnchorCap = 200;            // hard ceiling for the block

// Anchor registry entry: how often it appeared, where it last appeared, and
// whether it came from an intent-bearing (non-tool) message.
struct AnchorFact { int count; int lastIdx; bool valuable; };

}  // namespace

CompressPlan planCompression(const std::vector<std::string>& kinds,
                             const std::vector<std::string>& texts,
                             const CompressOptions& opts) {
    CompressPlan plan;
    const std::size_t n = texts.size();
    plan.messages.resize(n);
    if (n == 0) return plan;

    if (opts.budget <= 0) {  // no budget -> no policy; never invent one
        long long total = 0;
        for (std::size_t i = 0; i < n; i++) {
            plan.messages[i].action = Action::Keep;
            total += tok(texts[i]);
        }
        plan.stats.inTok = plan.stats.outTok = total;
        plan.stats.kept = static_cast<int>(n);
        return plan;
    }

    // Protection is known before pass 1: system prompt + newest `recent`.
    const std::size_t recent = opts.recent > 0 ? static_cast<std::size_t>(opts.recent) : 0;
    const std::size_t tailStart = n > recent ? n - recent : 0;

    // ---------- PASS 1: statistics ----------
    std::vector<long long> costArr(n, 0);
    std::vector<std::string> foldedArr(n);  // cached fold (empty = not folded)
    std::vector<double> valArr(n, kInf);
    std::vector<char> isSystem(n, 0);
    std::vector<char> protect(n, 0);

    std::unordered_map<std::string, AnchorFact> anchorMap;
    std::vector<std::string> anchorOrder;                    // insertion order
    std::unordered_map<std::string, int> recentAnchorCounts; // rolling window
    std::unordered_map<std::size_t, std::vector<std::string>> anchorsFor;  // per-message cache
    long long inTok = 0;

    for (std::size_t i = 0; i < n; i++) {
        const std::string& kind = kinds[i];
        const std::string& raw = texts[i];
        const bool sys = (kind == "system");
        const bool prot = sys || i >= tailStart;
        isSystem[i] = sys ? 1 : 0;
        protect[i] = prot ? 1 : 0;

        long long cost = tok(raw);
        if (!prot && kind == "tool" && cost > opts.toolFold) {
            foldedArr[i] = fold(raw, static_cast<std::size_t>(opts.head),
                                static_cast<std::size_t>(opts.tail));
            cost = tok(foldedArr[i]);
        }
        costArr[i] = cost;
        inTok += cost;

        if (!sys) {
            // One extraction per message, cached: the sliding window used to
            // re-scan the leaving message, doubling the regex work.
            auto [slot, inserted] = anchorsFor.try_emplace(i);
            if (inserted) {
                const std::size_t scanCap = (kind == "tool") ? kAnchorScanChars : kChatScanChars;
                const std::string scanned = raw.size() > scanCap ? raw.substr(0, scanCap) : raw;
                slot->second = extractAnchors(scanned, kAnchorCap);
            }
            const std::vector<std::string>& mine = slot->second;
            const bool valuable = (kind != "tool");
            int uniq = 0;
            for (const std::string& a : mine) {
                auto it = anchorMap.find(a);
                if (it == anchorMap.end()) {
                    anchorMap.emplace(a, AnchorFact{1, static_cast<int>(i), valuable});
                    anchorOrder.push_back(a);
                } else {
                    it->second.count++;
                    it->second.lastIdx = static_cast<int>(i);
                    it->second.valuable = it->second.valuable || valuable;
                }
                auto r = recentAnchorCounts.find(a);
                if (r == recentAnchorCounts.end() || r->second <= 0) uniq++;
            }
            double novel = (raw.find("⟨§") != std::string::npos) ? 0.6 : 1.0;
            valArr[i] = (1.0 + uniq * 0.5) * novel;

            // Slide the window: drop the message leaving it, add this one.
            if (i >= 40) {
                std::size_t leaving = i - 40;
                if (!isSystem[leaving]) {
                    auto cached = anchorsFor.find(leaving);
                    if (cached != anchorsFor.end()) {
                        for (const std::string& a : cached->second) {
                            auto r = recentAnchorCounts.find(a);
                            if (r != recentAnchorCounts.end()) r->second--;
                        }
                    }
                }
            }
            for (const std::string& a : mine) recentAnchorCounts[a]++;
        }
    }

    // ---------- anchor block: intent-bearing facts first, then newest ----------
    std::vector<std::pair<int, std::size_t>> ordered;  // (lastIdx, insertOrder)
    ordered.reserve(anchorOrder.size());
    for (std::size_t k = 0; k < anchorOrder.size(); k++) {
        ordered.emplace_back(anchorMap[anchorOrder[k]].lastIdx, k);
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [&anchorMap, &anchorOrder](const std::pair<int, std::size_t>& a,
                                                const std::pair<int, std::size_t>& b) {
                         const bool va = anchorMap[anchorOrder[a.second]].valuable;
                         const bool vb = anchorMap[anchorOrder[b.second]].valuable;
                         if (va != vb) return va;
                         return a.first > b.first;
                     });

    const long long anchorBudget = opts.budget / 7;  // ~15% of the budget
    long long anchorTok = 0;
    for (const auto& entry : ordered) {
        const std::string& a = anchorOrder[entry.second];
        plan.anchors.push_back(a);
        anchorTok += tok(a) + 1;
        if (plan.anchors.size() >= kAnchorCap) break;
        if (anchorTok > anchorBudget && plan.anchors.size() > 20) break;
    }

    // ---------- histogram over value -> (token sum, message count) ----------
    std::map<double, std::pair<long long, int>> bucket;
    for (std::size_t i = 0; i < n; i++) {
        if (protect[i]) continue;
        auto& b = bucket[valArr[i]];
        b.first += costArr[i];
        b.second++;
    }

    const long long targetBudget = opts.budget - anchorTok - 40;
    const long long need = inTok - targetBudget;
    auto cutFor = [&bucket](long long target) -> double {
        long long acc = 0;
        for (const auto& kv : bucket) {
            acc += kv.second.first;
            if (acc >= target) return kv.first;
        }
        return kInf;
    };

    double tombCut = -1.0;
    if (need > 0) tombCut = cutFor(need);

    double gistCut = -1.0;
    if (need > 0) {
        long long saved = 0;
        for (const auto& kv : bucket) if (kv.first <= tombCut) saved += kv.second.first;
        if (saved < need) {
            const long long rest = need - saved;
            long long acc = 0;
            for (auto it = bucket.rbegin(); it != bucket.rend(); ++it) {
                if (it->first <= tombCut) break;
                acc += it->second.first;
                if (acc >= rest) { gistCut = it->first; break; }
            }
        }
    }

    // ---------- PASS 2: emit the plan ----------
    long long outTok = anchorTok + 40;  // anchor block header overhead
    for (std::size_t i = 0; i < n; i++) {
        MessagePlan& m = plan.messages[i];
        if (protect[i]) {
            m.action = Action::Keep;
            outTok += tok(texts[i]);
            plan.stats.kept++;
            continue;
        }
        // A human turn is never blanked: prose intent has no anchor substitute,
        // so it degrades to a gist at worst.
        const bool canTomb = (kinds[i] != "user");
        if (valArr[i] <= tombCut && canTomb) {
            m.action = Action::Tomb;
            outTok += 2;
            plan.stats.tomb++;
            continue;
        }
        std::string text;
        const bool folded = !foldedArr[i].empty();
        if (folded) text = foldedArr[i];
        else text = texts[i];
        if (valArr[i] <= gistCut || (!canTomb && valArr[i] <= tombCut)) {
            text = "[要点] " + gist(text, static_cast<std::size_t>(opts.gistLen));
            plan.stats.gisted++;
        }
        if (folded) plan.stats.folded++;
        if (text == texts[i]) {
            m.action = Action::Keep;
        } else {
            m.action = Action::Rewrite;
            m.text = std::move(text);
        }
        outTok += tok(m.action == Action::Rewrite ? m.text : texts[i]);
        plan.stats.kept++;
    }

    plan.stats.inTok = inTok;
    plan.stats.outTok = outTok;
    plan.stats.anchors = static_cast<int>(plan.anchors.size());
    return plan;
}

}  // namespace cf
