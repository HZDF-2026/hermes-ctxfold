// main.cpp — line/length-framed worker protocol for host adapters.
//
// The host (Hermes plugin, and any future adapter) speaks this on stdin/stdout.
// No JSON: every record is "<tag>\t<byte-length>\n<raw UTF-8 bytes>", so the
// C++ side never parses or re-serializes a provider message. Message bodies
// pass through untouched unless the plan says Rewrite.
//
// Request header (first line):
//   CTXFOLD/1 compress <budget> <toolFold> <head> <tail> <recent> <gistLen> <n>
//   CTXFOLD/1 distill  <maxFacts> <maxLessons> <maxDecisions>
//   CTXFOLD/1 id       <topicBytes> <textBytes>
//   CTXFOLD/1 ping
//
// Every response starts with "CTXFOLD/1 OK ..." and a failure with
// "CTXFOLD/1 ERR ..." on stdout (exit code 1).
//
// compress response:
//   CTXFOLD/1 OK <kept> <tomb> <gisted> <folded> <inTok> <outTok> <anchorCount>
//   then n action records: "keep\n" | "tomb\n" | "rewrite\t<len>\n<bytes>"
//   then <anchorCount> records: "A\t<len>\n<bytes>"
//
// distill response:
//   CTXFOLD/1 OK <facts> <lessons> <decisions>
//   then records tagged F / L / D: "<tag>\t<len>\n<bytes>"
//
// id response: "CTXFOLD/1 OK <12-hex-id>";  ping: "CTXFOLD/1 OK pong"
//
// One request per process invocation is the supported mode (stateless, no
// shared state between host sessions). The loop is kept so a future persistent
// worker needs no protocol change.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "compress.h"
#include "core.h"

namespace {

void die(const std::string& message) {
    std::cout << "ERR " << message << "\n";
    std::cout.flush();
    std::exit(1);
}

std::string readAllStdin() {
    std::string data;
    char buf[8192];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) data.append(buf, n);
    return data;
}

// Cursor over the request bytes: exact-length reads, no allocation churn.
struct Reader {
    const std::string& src;
    std::size_t pos = 0;
    explicit Reader(const std::string& s) : src(s) {}

    bool eof() const { return pos >= src.size(); }

    std::string line() {
        std::size_t nl = src.find('\n', pos);
        if (nl == std::string::npos) {
            std::string tail = src.substr(pos);
            pos = src.size();
            return tail;
        }
        std::string out = src.substr(pos, nl - pos);
        pos = nl + 1;
        return out;
    }

    std::string bytes(std::size_t len) {
        if (pos + len > src.size()) die("truncated record");
        std::string out = src.substr(pos, len);
        pos += len;
        return out;
    }
};

std::vector<std::string> splitWs(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
        if (i >= s.size()) break;
        std::size_t start = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\t') i++;
        out.push_back(s.substr(start, i - start));
    }
    return out;
}

long long toLL(const std::vector<std::string>& f, std::size_t i, long long dflt) {
    if (i >= f.size() || f[i].empty()) return dflt;
    return std::atoll(f[i].c_str());
}

std::string header() {
    std::string h;
    h.reserve(64);
    h += "CTXFOLD/1 ";
    return h;
}

void emit(std::string& out, const std::string& tag, const std::string& body) {
    out += tag;
    out += "\t" + std::to_string(body.size()) + "\n";
    out += body;
}

void handleCompress(const std::vector<std::string>& f, Reader& r, std::string& out) {
    // f[0] = "CTXFOLD/1", f[1] = "compress"
    if (f.size() < 9) die("compress: expected 7 arguments after the op name");
    cf::CompressOptions opts;
    opts.budget = toLL(f, 2, 0);
    opts.toolFold = static_cast<int>(toLL(f, 3, 400));
    opts.head = static_cast<int>(toLL(f, 4, 6));
    opts.tail = static_cast<int>(toLL(f, 5, 4));
    opts.recent = static_cast<int>(toLL(f, 6, 20));
    opts.gistLen = static_cast<int>(toLL(f, 7, 120));
    const long long n = toLL(f, 8, 0);
    if (n < 0 || n > 200000) die("compress: bad message count");

    std::vector<std::string> kinds;
    std::vector<std::string> texts;
    kinds.reserve(static_cast<std::size_t>(n));
    texts.reserve(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; i++) {
        std::vector<std::string> head = splitWs(r.line());
        if (head.size() != 2) die("compress: malformed message header");
        kinds.push_back(head[0]);
        texts.push_back(r.bytes(static_cast<std::size_t>(std::atoll(head[1].c_str()))));
    }

    cf::CompressPlan plan = cf::planCompression(kinds, texts, opts);
    out += header() + "OK " + std::to_string(plan.stats.kept) + " " +
           std::to_string(plan.stats.tomb) + " " + std::to_string(plan.stats.gisted) + " " +
           std::to_string(plan.stats.folded) + " " + std::to_string(plan.stats.inTok) + " " +
           std::to_string(plan.stats.outTok) + " " + std::to_string(plan.anchors.size()) + "\n";
    for (const cf::MessagePlan& m : plan.messages) {
        if (m.action == cf::Action::Keep) out += "keep\n";
        else if (m.action == cf::Action::Tomb) out += "tomb\n";
        else emit(out, "rewrite", m.text);
    }
    for (const std::string& a : plan.anchors) emit(out, "A", a);
}

void handleDistill(const std::vector<std::string>& f, Reader& r, std::string& out) {
    if (f.size() < 6) die("distill: expected maxFacts maxLessons maxDecisions <len>");
    std::size_t maxFacts = static_cast<std::size_t>(toLL(f, 2, 40));
    std::size_t maxLessons = static_cast<std::size_t>(toLL(f, 3, 12));
    std::size_t maxDecisions = static_cast<std::size_t>(toLL(f, 4, 12));
    std::string text = r.bytes(static_cast<std::size_t>(toLL(f, 5, 0)));
    cf::Distilled d = cf::distillText(text, maxFacts, maxLessons, maxDecisions);
    out += header() + "OK " + std::to_string(d.facts.size()) + " " +
           std::to_string(d.lessons.size()) + " " + std::to_string(d.decisions.size()) + "\n";
    for (const std::string& s : d.facts) emit(out, "F", s);
    for (const std::string& s : d.lessons) emit(out, "L", s);
    for (const std::string& s : d.decisions) emit(out, "D", s);
}

void handleId(const std::vector<std::string>& f, Reader& r, std::string& out) {
    if (f.size() < 4) die("id: expected <topicBytes> <textBytes>");
    std::string topic = r.bytes(static_cast<std::size_t>(toLL(f, 2, 0)));
    std::string text = r.bytes(static_cast<std::size_t>(toLL(f, 3, 0)));
    out += header() + "OK " + cf::entryId(topic, text) + "\n";
}

}  // namespace

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::string input = readAllStdin();
    Reader r(input);
    std::string out;
    while (!r.eof()) {
        std::string head = r.line();
        if (head.empty()) continue;
        if (head.compare(0, 10, "CTXFOLD/1 ") != 0) die("bad protocol header");
        std::vector<std::string> f = splitWs(head);
        if (f.size() < 2) die("missing op name");
        const std::string& op = f[1];
        if (op == "ping") { out += header() + "OK pong\n"; continue; }
        if (op == "compress") { handleCompress(f, r, out); continue; }
        if (op == "distill") { handleDistill(f, r, out); continue; }
        if (op == "id") { handleId(f, r, out); continue; }
        die("unknown op: " + op);
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    return 0;
}
