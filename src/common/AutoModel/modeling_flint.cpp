/// \file modeling_flint.cpp
/// \brief Flint on the open kernels (open_qwen36::Engine, kernel set export_flint_kernels.py)

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "AutoModel/modeling_flint.hpp"
#include "metrices.hpp"

namespace {
// the turn tokens a system part ends at: <|user|>, <|assistant|>, <|observation|>, <|developer|>
constexpr int kFlintUser = 150003, kFlintAssistant = 150004, kFlintObservation = 150005, kFlintDeveloper = 150018;

std::string trim_ws(std::string v) {
    while (!v.empty() && (v.front() == '\n' || v.front() == '\r' || v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ' || v.back() == '\t')) v.pop_back();
    return v;
}

// One tool call's body: name<arg_key>k</arg_key><arg_value>v</arg_value>... A value parses as JSON when it
// can (the template writes non-strings with tojson) and stays a string otherwise.
std::pair<std::string, nlohmann::json> parse_flint_call(const std::string& block) {
    const std::string kopen = "<arg_key>", kclose = "</arg_key>", vopen = "<arg_value>", vclose = "</arg_value>";
    nlohmann::json args = nlohmann::json::object();
    const size_t first = block.find(kopen);
    std::string name = trim_ws(block.substr(0, first));
    size_t pos = first;
    while (pos != std::string::npos) {
        const size_t ks = block.find(kopen, pos);
        if (ks == std::string::npos) break;
        const size_t ke = block.find(kclose, ks + kopen.size());
        if (ke == std::string::npos) break;
        const std::string key = trim_ws(block.substr(ks + kopen.size(), ke - ks - kopen.size()));
        const size_t vs = block.find(vopen, ke + kclose.size());
        if (vs == std::string::npos) break;
        size_t ve = block.find(vclose, vs + vopen.size());
        const size_t next_key = block.find(kopen, vs + vopen.size());
        if (ve == std::string::npos || (next_key != std::string::npos && next_key < ve)) ve = next_key;
        if (ve == std::string::npos) ve = block.size();
        const std::string val = block.substr(vs + vopen.size(), ve - vs - vopen.size());
        try {
            args[key] = nlohmann::json::parse(val);
        } catch (...) {
            args[key] = val;
        }
        pos = ve;
    }
    return {name, args};
}
}  // namespace

Flint::Flint(oflm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Flint") {}

void Flint::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    // the open kernels only (there is no closed Flint engine)
    auto open_engine = this->_shared_select_open_engine("OFLM_FLINT_ENGINE", "Flint");
    if (!open_engine) throw std::runtime_error("Flint runs on the open kernels only, and none were found for " + model_path);
    this->lm_engine = std::move(open_engine);
    this->lm_engine->clear_context();
    this->_shared_setup_tokenizer(model_path);
    this->sampler.reset();

    // generation_config.json's
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.95;
    config.min_p = 0.0;
    config.temperature = 1.0;
    config.rep_penalty = 1.0;
    config.freq_penalty = 0.0;
    config.pre_penalty = 0.0;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) this->profiler_list[i].reset();
}

std::string Flint::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    // The template writes a tool call as <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>, which minja's
    // capability probe does not recognise (it looks for "argument_needle":), so its polyfills rewrote the history's
    // tool calls and tool results as JSON text, ids included, and Flint copied that format in its next turn. The
    // template renders both itself; its arguments.items() needs objects, so string arguments are parsed here.
    minja::chat_template_options opt;
    opt.polyfill_tool_calls = false;
    opt.polyfill_tool_responses = false;
    opt.polyfill_object_arguments = false;
    nlohmann::ordered_json msgs = messages;
    for (auto& m : msgs) {
        if (!m.is_object() || !m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
        for (auto& tc : m["tool_calls"]) {
            auto& fn = tc.contains("function") ? tc["function"] : tc;
            if (!fn.contains("arguments") || !fn["arguments"].is_string()) continue;
            const std::string a = fn["arguments"].get<std::string>();
            try {
                fn["arguments"] = nlohmann::ordered_json::parse(a.empty() ? "{}" : a);
            } catch (...) {
                fn["arguments"] = nlohmann::ordered_json{{"input", a}};
            }
            if (!fn["arguments"].is_object()) fn["arguments"] = nlohmann::ordered_json{{"input", fn["arguments"]}};
        }
    }
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = msgs;
    inputs.extra_context = this->extra_context;
    if (!tools.empty() && this->enable_tool) inputs.tools = tools;
    return this->_shared_apply_template(inputs, opt);
}

bool Flint::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    std::string templated_text;
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools);
    } else {
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        if (!this->user_system_prompt.empty())
            messages.push_back({{"role", "system"}, {"content", this->user_system_prompt}});
        messages.push_back({{"role", "user"}, {"content", input.prompt}});
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    if (meta_info.restore_allowed) {
        this->total_tokens = this->lm_engine->restore();
        this->token_history = checkpoint_his;
    }

    // A long system part (the system message and the tools) shared by requests that differ after it:
    // the engine keeps the state at its end, and a request with the same system tokens restores it
    // and prefills only the rest (OFLM_PREFIX_CACHE_MIN tokens and up, default 512; 0 off).
    static const size_t prefix_min = [] {
        const char* e = std::getenv("OFLM_PREFIX_CACHE_MIN");
        return e ? static_cast<size_t>(std::strtoul(e, nullptr, 10)) : size_t{512};
    }();
    double pre_ns = 0;
    int pre_tokens = 0;
    if (!meta_info.restore_allowed && prefix_min > 0 && !input.messages.empty() &&
        input.messages[0].value("role", "") == "system" && tokens.size() > 2) {
        size_t b = 0;
        for (size_t i = 1; i < tokens.size() && !b; ++i)
            if (tokens[i] == kFlintUser || tokens[i] == kFlintAssistant || tokens[i] == kFlintObservation || tokens[i] == kFlintDeveloper)
                b = i;
        if (b >= prefix_min) {
            const bool same = prefix_his_.size() == b && std::equal(prefix_his_.begin(), prefix_his_.end(), tokens.begin());
            if (same && this->lm_engine->restore_prefix() == static_cast<int>(b)) {
                this->total_tokens = static_cast<int>(b);
                this->token_history = prefix_his_;
                this->checkpoint_his = prefix_his_;
                header_print("OFLM", "System prompt cache hit: " << b << " tokens restored, " << tokens.size() - b << " to prefill");
            } else if (this->token_history.empty()) {
                std::vector<int> pre(tokens.begin(), tokens.begin() + b);
                if (!this->_shared_insert(meta_info, pre, is_cancelled, nullptr)) return false;
                pre_ns = static_cast<double>(meta_info.prefill_duration);
                pre_tokens = meta_info.prompt_tokens;
                if (this->lm_engine->checkpoint_prefix() == static_cast<int>(b)) {
                    prefix_his_ = this->token_history;
                    header_print("OFLM", "System prompt cached: " << b << " tokens");
                } else {
                    prefix_his_.clear();
                }
            }
        }
    }

    const bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
    if (pre_tokens) {   // report the whole prompt's prefill
        meta_info.prefill_duration += static_cast<uint64_t>(pre_ns);
        meta_info.prompt_tokens += pre_tokens;
    }
    checkpoint_his = token_history;
    this->lm_engine->checkpoint();
    return success;
}

std::string Flint::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Flint::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (!this->insert(meta_info, input, is_cancelled)) return "";
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

NonStreamResult Flint::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;
    const std::string start_tag = "<tool_call>", end_tag = "</tool_call>";
    // the reasoning is not content
    std::string text = response_text;
    const size_t te = text.find("</think>");
    if (te != std::string::npos) {
        const size_t ts = text.find("<think>");
        result.reasoning_content = trim_ws(text.substr(ts == std::string::npos ? 0 : ts + 7, te - (ts == std::string::npos ? 0 : ts + 7)));
        text = text.substr(te + 8);
    }
    size_t from = 0;
    while (true) {
        const size_t s = text.find(start_tag, from);
        if (s == std::string::npos) break;
        const size_t bs = s + start_tag.size();
        size_t e = text.find(end_tag, bs);
        const size_t be = e == std::string::npos ? text.size() : e;
        from = e == std::string::npos ? text.size() : e + end_tag.size();
        auto [name, args] = parse_flint_call(text.substr(bs, be - bs));
        result.tool_calls_list.emplace_back(name, args.dump());
    }
    if (result.tool_calls_list.empty()) {
        result.content = trim_ws(text);
    } else {
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
        const size_t first = text.find(start_tag);
        if (first != std::string::npos && first > 0) result.content = trim_ws(text.substr(0, first));
    }
    return result;
}

StreamResult Flint::parse_stream_content(const std::string content) { return parse_stream_content_impl(content, false); }

StreamResult Flint::parse_stream_content_final(const std::string content) { return parse_stream_content_impl(content, true); }

StreamResult Flint::parse_stream_content_impl(const std::string content, bool is_final) {
    const std::string THINK_START = "<think>", THINK_END = "</think>", TOOL_START = "<tool_call>", TOOL_END = "</tool_call>";
    StreamResult result;
    buffer_ += content;
    while (true) {
        if (!is_in_tool_block_) {
            const size_t stray = buffer_.find(TOOL_END);
            if (stray != std::string::npos) buffer_.erase(stray, TOOL_END.size());
            const size_t ts = buffer_.find(TOOL_START);
            if (ts != std::string::npos && current_mode_ == StreamEventType::CONTENT) {
                if (ts > 0) {
                    result.content = buffer_.substr(0, ts);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(ts);
                    return result;
                }
                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(TOOL_START.size());
                result.type = StreamEventType::WAITING;
                return result;
            }
        }
        if (is_in_tool_block_) {
            const size_t te = buffer_.find(TOOL_END);
            if (te != std::string::npos || (is_final && !buffer_.empty())) {
                const size_t end = te == std::string::npos ? buffer_.size() : te;
                const std::string block = buffer_.substr(0, end);
                buffer_ = te == std::string::npos ? std::string() : buffer_.substr(te + TOOL_END.size());
                is_in_tool_block_ = false;
                try {
                    auto [name, args] = parse_flint_call(block);
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));
                    result.tool_name = name;
                    result.tool_args_str = args.dump();
                } catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = "[Error parsing tool call]";
                }
                return result;
            }
            result.type = StreamEventType::WAITING;
            return result;
        }
        if (current_mode_ == StreamEventType::CONTENT) {
            const size_t p = buffer_.find(THINK_START);
            if (p != std::string::npos) {
                if (p > 0) {
                    result.content = buffer_.substr(0, p);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(p);
                    return result;
                }
                buffer_ = buffer_.substr(THINK_START.size());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        } else if (current_mode_ == StreamEventType::REASONING) {
            const size_t p = buffer_.find(THINK_END);
            if (p != std::string::npos) {
                if (p > 0) {
                    result.content = buffer_.substr(0, p);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(p);
                    return result;
                }
                buffer_ = buffer_.substr(THINK_END.size());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }
        if (!buffer_.empty()) {
            // a '<' near the end may start a tag: hold it back
            const size_t lt = buffer_.rfind('<');
            if (!is_final && lt != std::string::npos && buffer_.size() - lt <= 12) {
                if (lt > 0) {
                    result.content = buffer_.substr(0, lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(lt);
                    return result;
                }
                result.type = StreamEventType::WAITING;
                return result;
            }
            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }
        break;
    }
    result.type = current_mode_;
    return result;
}
