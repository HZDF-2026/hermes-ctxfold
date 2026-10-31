// core.cpp — deterministic primitives. See core.h for the contract.
//
// Fidelity notes (deliberately preserved JS quirks):
//  - the Windows-path pattern also matches "s://..." inside URLs (a drive
//    letter followed by ':' + '/'), so URL fragments surface as anchors —
//    harmless (verbatim, still exact) and required for cross-engine parity;
//  - anchor order is per-regex in engine order, first-seen kept (Set
//    semantics), no substring dedup at this layer;
//  - `\b` is ASCII-only in JS, so the regexes that use it are compiled with
//    re::ASCII-equivalent semantics (see the raw byte checks in collect()).
#include "core.h"

#include <algorithm>
#include <regex>
#include <unordered_map>
#include <unordered_set>

#include "sha256.h"

namespace cf {
namespace {

std::size_t cpLen(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    return 4;
}

// Decode one code point (best effort; invalid bytes decode as themselves).
unsigned decode(const std::string& s, std::size_t i, std::size_t& len) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    len = cpLen(b);
    if (len == 1 || i + len > s.size()) return b;
    unsigned cp = 0;
    if (len == 2) cp = b & 0x1F;
    else if (len == 3) cp = b & 0x0F;
    else cp = b & 0x07;
    for (std::size_t k = 1; k < len && i + k < s.size(); k++) {
        unsigned char c = static_cast<unsigned char>(s[i + k]);
        if ((c & 0xC0) != 0x80) { len = k; return cp; }
        cp = (cp << 6) | (c & 0x3F);
    }
    return cp;
}

std::size_t codePoints(const std::string& s) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < s.size();) { std::size_t l; decode(s, i, l); i += l; n++; }
    return n;
}

std::size_t cpOffset(const std::string& s, std::size_t maxCp) {
    std::size_t i = 0, n = 0;
    while (i < s.size() && n < maxCp) { std::size_t l; decode(s, i, l); i += l; n++; }
    return i;
}

std::string trimAscii(const std::string& s) {
    std::size_t a = 0, b = s.size();
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'
                              || c == '\v' || c == '\f'; };
    while (a < b && ws(s[a])) a++;
    while (b > a && ws(s[b - 1])) b--;
    return s.substr(a, b - a);
}

bool isAsciiWs(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\v'
                            || c == '\f'; }

// Split on '\n', keeping JS split() semantics (a trailing newline yields a
// trailing empty segment).
std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == '\n') { out.push_back(s.substr(start, i - start)); start = i + 1; }
    }
    return out;
}

// Collect up to `cap` matches of `rx`, first-seen-wins, insertion order.
// Membership uses a set: the previous linear scan made dense outputs
// quadratic, which dominated the whole pass on tool-heavy transcripts.
void collect(const std::regex& rx, const std::string& text, std::size_t cap,
             std::vector<std::string>& out) {
    static thread_local std::unordered_set<std::string> seen;
    if (out.empty()) seen.clear();
    std::size_t n = 0;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), rx);
         it != std::sregex_iterator() && n < cap; ++it, ++n) {
        std::string m = trimAscii(it->str());
        if (m.empty()) continue;
        if (seen.insert(m).second) out.push_back(m);
    }
}

void collectGroup1(const std::regex& rx, const std::string& text, std::size_t cap,
                   std::vector<std::string>& out) {
    std::size_t n = 0;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), rx);
         it != std::sregex_iterator() && n < cap; ++it, ++n) {
        std::string m = trimAscii((*it)[1].str());
        if (m.empty()) continue;
        if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
    }
}

bool endsWithTerminator(const std::string& s, std::size_t i, std::size_t& len) {
    // 。 U+3002 = E3 80 82；！ U+FF01 = EF BC 81；？ U+FF1F = EF BC 9F
    if (i + 3 <= s.size()) {
        unsigned char b0 = s[i], b1 = s[i + 1], b2 = s[i + 2];
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) { len = 3; return true; }
        if (b0 == 0xEF && b1 == 0xBC && (b2 == 0x81 || b2 == 0x9F)) { len = 3; return true; }
    }
    if (s[i] == '!' || s[i] == '?') { len = 1; return true; }
    return false;
}

std::vector<std::string> splitSentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (std::size_t i = 0; i < text.size();) {
        std::size_t tlen = 0;
        if (endsWithTerminator(text, i, tlen)) {
            cur.append(text, i, tlen);
            i += tlen;
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            continue;
        }
        if (text[i] == '.') {
            // '.' ends a sentence only before whitespace (consumed) — file
            // names like babele.exe must survive intact.
            if (i + 1 < text.size() && isAsciiWs(text[i + 1])) {
                cur += '.';
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
                i += 2;
                continue;
            }
            cur += '.'; i += 1; continue;
        }
        if (text[i] == '\n') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            i += 1; continue;
        }
        std::size_t l; decode(text, i, l);
        cur.append(text, i, std::min(l, text.size() - i));
        i += l;
    }
    if (!cur.empty()) out.push_back(cur);
    std::vector<std::string> kept;
    for (auto& s : out) {
        std::string t = trimAscii(s);
        if (codePoints(t) > 8) kept.push_back(t);
    }
    return kept;
}

// JS: text.toLowerCase().match(/[a-z0-9\u4e00-\u9fa5]{2,}/g) — runs of
// lowercase ASCII letters, digits, or CJK, at least two characters long.
// A run may mix classes (the JS character class is one set).
std::vector<std::string> tokenRuns(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (std::size_t i = 0; i < text.size();) {
        std::size_t l; unsigned cp = decode(text, i, l);
        bool inClass = false;
        if (l == 1) {
            char c = static_cast<char>(cp);
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) { cur += c; inClass = true; }
        } else if (cp >= 0x4E00 && cp <= 0x9FA5) {
            cur.append(text, i, l); inClass = true;
        }
        if (!inClass) {
            if (cur.size() >= 2) out.push_back(cur);
            cur.clear();
        }
        i += l;
    }
    if (cur.size() >= 2) out.push_back(cur);
    return out;
}

}  // namespace

long long tok(const std::string& text) {
    double n = 0;
    for (std::size_t i = 0; i < text.size();) {
        std::size_t l; unsigned cp = decode(text, i, l);
        if (cp == ' ' || cp == '\n') { i += l; continue; }
        if (cp > 0x2E7F) n += 1.0;
        else n += 0.25;
        i += l;
    }
    long long whole = static_cast<long long>(n);
    long long up = (n > static_cast<double>(whole)) ? 1 : 0;
    return whole + up;
}

std::string normText(const std::string& text) {
    std::string out;
    bool ws = false;
    for (std::size_t i = 0; i < text.size();) {
        char c = text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
            ws = !out.empty();
            i += 1;
            continue;
        }
        if (ws && !out.empty()) out += ' ';
        ws = false;
        std::size_t l = cpLen(static_cast<unsigned char>(c));
        out.append(text, i, std::min(l, text.size() - i));
        i += l;
    }
    return out;
}

std::vector<std::string> extractAnchors(const std::string& text, std::size_t limit) {
    std::vector<std::string> out;
    // Compiled once per process: std::regex construction dominates otherwise.
    static const std::regex rxWinPath(R"([A-Za-z]:[\\/][^\s"'。）)】\]，,；;]+)",
                                      std::regex::icase | std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxPosixPath(R"((?:\/[A-Za-z0-9_.-]+){2,})",
                                        std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxUrl(R"(https?:\/\/[^\s"'。）)】\]，,；;]+)",
                                  std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxHash(R"(\b[0-9a-f]{40,64}\b)",
                                   std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxNumber(R"(\b\d[\d,.]{2,}(?:\s?(?:GB|MB|KB|TB|k|ms|s|min|hour|天|小时|分钟|秒|元|美元|%)|\s*%(?!\w))?)",
                                     std::regex::icase | std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxCommand(R"(\b(?:gh|npm|pip|node|qv|docker|git)\s+\S+)",
                                      std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxDotted(R"(\b[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*){1,5}\b)",
                                     std::regex::ECMAScript | std::regex::optimize);
    static const std::regex rxBacktick(R"(`([^`\n]{2,160})`)",
                                       std::regex::ECMAScript | std::regex::optimize);

    collect(rxWinPath, text, 40, out);
    collect(rxPosixPath, text, 40, out);
    collect(rxUrl, text, 40, out);
    collect(rxHash, text, 40, out);
    collect(rxNumber, text, 40, out);
    collect(rxCommand, text, 40, out);
    collect(rxDotted, text, 40, out);
    collectGroup1(rxBacktick, text, 40, out);
    if (out.size() > limit) out.resize(limit);
    return out;
}

std::string fold(const std::string& text, std::size_t keepHead,
                 std::size_t keepTail) {
    std::vector<std::string> lines = splitLines(text);
    if (lines.size() <= keepHead + keepTail + 2) return text;
    std::vector<std::string> mid;
    for (std::size_t i = keepHead; i < lines.size() - keepTail; i++) mid.push_back(lines[i]);
    std::string midJoined;
    for (std::size_t i = 0; i < mid.size(); i++) { if (i) midJoined += '\n'; midJoined += mid[i]; }
    std::string h = gr::sha256Hex(midJoined).substr(0, 16);
    std::string head;
    for (std::size_t i = 0; i < keepHead; i++) { if (i) head += '\n'; head += lines[i]; }
    std::string marker = "[…ctxfold: 折叠 " + std::to_string(mid.size()) +
                         " 行 (sha256:" + h + ")…]";
    std::string tail;
    std::size_t from = lines.size() - keepTail;
    for (std::size_t i = from; i < lines.size(); i++) {
        if (i > from) tail += '\n';
        tail += lines[i];
    }
    return head + "\n" + marker + "\n" + tail;
}

std::string gist(const std::string& text, std::size_t maxLen) {
    std::string t = normText(text);
    std::size_t cut = cpOffset(t, maxLen);
    std::string head = t.substr(0, cut);
    if (codePoints(t) > maxLen) {
        head += " …[裁断, 全文 " + std::to_string(tok(t)) + " tok]";
    }
    return head;
}

Distilled distillText(const std::string& text, std::size_t maxFacts,
                      std::size_t maxLessons, std::size_t maxDecisions) {
    Distilled d;
    d.facts = extractAnchors(normText(text), maxFacts);
    static const std::regex lesson(
        R"((失败|报错|错误|教训|踩坑|注意|避免|根因|bug|fix(?:ed)?|lesson|mistake|pitfall|don't))",
        std::regex::icase | std::regex::ECMAScript);
    static const std::regex decision(
        R"((决定|决策|选定|方案是|拍板|采用|改用|decided|we (?:use|go with)|chose))",
        std::regex::icase | std::regex::ECMAScript);
    for (const std::string& s : splitSentences(text)) {
        if (d.lessons.size() < maxLessons && std::regex_search(s, lesson))
            d.lessons.push_back(gist(s, 240));
        else if (d.decisions.size() < maxDecisions && std::regex_search(s, decision))
            d.decisions.push_back(gist(s, 240));
    }
    return d;
}

std::string entryId(const std::string& topic, const std::string& text) {
    std::string payload = topic + "\n" + normText(text);
    return gr::sha256Hex(payload).substr(0, 12);
}

double recallScore(const std::string& query, const std::string& entryText,
                   const std::string& entryTopic,
                   const std::vector<std::string>& entryAnchors, int hits,
                   const std::string& kind) {
    std::vector<std::string> qa = extractAnchors(normText(query), 200);
    std::unordered_set<std::string> qaSet(qa.begin(), qa.end());
    double score = 0;
    for (const std::string& a : entryAnchors) {
        if (qaSet.count(a)) score += 3;
    }
    std::vector<std::string> qt = tokenRuns(query);
    std::unordered_set<std::string> qtSet(qt.begin(), qt.end());
    std::vector<std::string> et = tokenRuns(entryText + " " + entryTopic);
    std::unordered_set<std::string> etSet(et.begin(), et.end());
    std::size_t overlap = 0;
    for (const std::string& t : qtSet) {
        if (etSet.count(t)) overlap++;
    }
    score += static_cast<double>(overlap) / (1.0 + static_cast<double>(etSet.size()) / 32.0);
    score += 0.15 * static_cast<double>(hits);
    if (kind == "lesson") score += 0.3;
    return score;
}

}  // namespace cf
