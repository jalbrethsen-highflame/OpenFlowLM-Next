/// \file modeling_flint.hpp
/// \brief Flint (RecursiveSI/flint, FlintForCausalLM) on the open kernels
/// \note Text only. The chat format is <|system|> / <|user|> / <|assistant|> / <|observation|> turns
///       (chat_template.jinja); the model reasons in <think>...</think> and calls tools as
///       <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>.

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

/************              Flint            **************/
class Flint : public AutoModel {
private:
    bool enable_tool = true;
    /// the system prompt's tokens whose state the engine holds as its prefix snapshot
    std::vector<int> prefix_his_;

public:
    Flint(oflm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;

private:
    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

public:
    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        // Flint always reasons first (its template has no switch for it)
        return parameter_name == "enable_think" || parameter_name == "reasoning_effort" || parameter_name == "toggle_think";
    }
};
