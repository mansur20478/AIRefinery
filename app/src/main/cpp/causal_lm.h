#pragma once
#include <string>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <llama.h>
#include <android/log.h>

#define LOG_TAG "CausalLM"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct GenerationParams {
    int maxTokens;
    int topK;
    float topP;
    float temperature;
    bool useGreedy;

    int n_threads;
};

struct CausalLMConfig {
    std::string modelPath;
};

using TokenCallback = std::function<bool(const std::string& token)>;
using DoneCallback = std::function<void()>;

struct GenerationFuncArgs {
    std::string prompt;
    GenerationParams params;
    TokenCallback token_callback;
    DoneCallback done_callback;
};

class CausalLM {
public:
    CausalLM();
    ~CausalLM();

    bool init(const CausalLMConfig& config);
    bool generate(const std::string& prompt, GenerationParams params, TokenCallback token_callback, DoneCallback done_callback);
    void stop_generation();
private:
    std::vector<llama_token> tokenize(const std::string& prompt, int max_tokens);
    std::string detokenize_one(llama_token token);

    bool setup_context(const GenerationParams& params);
    void free_context();

    bool setup_sampler(const GenerationParams& params);
    void free_sampler();

    llama_token decode(llama_token *tokens, int n_tokens);
    bool generate_impl(const std::string& prompt, GenerationParams params, TokenCallback token_callback);
    void generate_worker_loop();

    llama_context *ctx = nullptr;
    llama_model *model = nullptr;
    llama_sampler *smpl = nullptr;
    const llama_vocab *vocab = nullptr;

    bool should_stop;

    bool should_terminate;
    std::condition_variable new_job_condition;
    std::mutex jobs_queue_mutex;
    std::queue<GenerationFuncArgs> jobs;
    std::unique_ptr<std::thread> worker_thread;
};