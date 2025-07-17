#include <vector>
#include <algorithm>
#include <mutex>
#include <unistd.h>
#include <llama.h>
#include "causal_lm.h"

CausalLM::CausalLM() {
}

CausalLM::~CausalLM() {
    stop_generation();
    should_terminate = true;
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

bool CausalLM::init(const CausalLMConfig& config) {
    should_stop = false;
    worker_thread = nullptr;

    model = llama_model_load_from_file(config.modelPath.c_str(), llama_model_default_params());
    if (!model) {
        LOG_ERROR("Failed to load model");
        return false;
    }
    vocab = llama_model_get_vocab(model);
    worker_thread = std::make_unique<std::thread>([=](){
        should_terminate = false;
        generate_worker_loop();
    });
    LOG_INFO("Initialized model");
    return true;
}

bool CausalLM::setup_context(const GenerationParams& params) {
    if (ctx != nullptr)
        free_context();
    
    // setup context params
    llama_context_params ctx_params = llama_context_default_params();
    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    if (params.n_threads != -1) {
        n_threads = params.n_threads;
    }
    LOG_INFO("Default number of threads is %d, changing to %d threads", ctx_params.n_threads, n_threads);

    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_ctx = params.maxTokens;
    ctx = llama_init_from_model(model, ctx_params);
    return ctx != nullptr;
}

void CausalLM::free_context() {
    if (ctx == nullptr)
        return;
    llama_free(ctx);
    ctx = nullptr;
}

bool CausalLM::setup_sampler(const GenerationParams& params) {
    if (smpl)
        free_sampler();

    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);
    if (params.useGreedy) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }
    else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }
    return true;
}

void CausalLM::free_sampler() {
    if (!smpl)
        return;
    llama_sampler_free(smpl);
    smpl = nullptr;
}

std::vector<llama_token> CausalLM::tokenize(const std::string& prompt, int max_tokens) {
    std::vector<llama_token> prompt_tokens(max_tokens);
    int n_prompt_tokens = llama_tokenize(
            vocab, prompt.c_str(), (int)prompt.size(),
            prompt_tokens.data(), max_tokens, true, false);
    prompt_tokens.resize(n_prompt_tokens);
    return prompt_tokens;
}

std::string CausalLM::detokenize_one(llama_token token) {
    const int TEXT_MAX_LEN = 128;
    char token_c_str[TEXT_MAX_LEN];
    int len = llama_detokenize(vocab, &token, 1, token_c_str, TEXT_MAX_LEN, false, true);
    return std::string(token_c_str, len);
}

bool CausalLM::generate_impl(const std::string& prompt, GenerationParams params,
                        TokenCallback token_callback) {
    if (!setup_context(params)) {
        LOG_ERROR("Failed to setup context");
        return false;
    }
    if (!setup_sampler(params)) {
        LOG_ERROR("Failed to setup sampler");
        return false;
    }

    std::vector<llama_token> prompt_tokens = tokenize(prompt, params.maxTokens);
    
    should_stop = false;
    if (prompt_tokens.size() == 0) {
        LOG_ERROR("Failed to tokenize text");
        return false;
    }
    if (prompt_tokens.size() >= params.maxTokens) {
        LOG_ERROR("Number of prompt tokens exceed max tokens");
        return false;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token next_token = 0;

    float ttft = -1;
    float ttlt = -1;
    int num_decode_iter = 0;

    for (int n_past = 0; n_past + batch.n_tokens < params.maxTokens && !should_stop; ) {
        const auto start_time = ggml_time_us(); 
        int ret_code = llama_decode(ctx, batch);
        if (ret_code) {
            LOG_ERROR("Failed to llama_decode with code %d", ret_code);
            return false;
        }
        const auto end_time = ggml_time_us();
        
        if (n_past == 0) {
            ttft = (end_time - start_time) / 1000000.0;
            ttlt = ttft;
        }
        else {
            ttlt += (end_time - start_time) / 1000000.0;
            num_decode_iter += 1;
        }

        n_past += batch.n_tokens;

        next_token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }
        token_callback(detokenize_one(next_token));
        batch = llama_batch_get_one(&next_token, 1);
    }
    
    float decode_latency = (ttlt - ttft) / (num_decode_iter == 0 ? 1: num_decode_iter); 
    LOG_INFO("Time to first token: %f seconds, Time to last token %f, Decode latency %f", ttft, ttlt, decode_latency);

    float prefill_speed = static_cast<int>(prompt_tokens.size()) / ttft;
    float decode_speed = 1 / decode_latency;
    LOG_INFO("Average prefill phase %f tokens/second, decode phase %f tokens/second", prefill_speed, decode_speed);
    
    free_sampler();
    free_context();
    return true;
}

bool CausalLM::generate(const std::string& prompt, GenerationParams params,
                        TokenCallback token_callback, DoneCallback done_callback) {
    {
        std::unique_lock<std::mutex> lock(jobs_queue_mutex);
        jobs.push({prompt, params, token_callback, done_callback});
    }
    new_job_condition.notify_all();
    return true;
}

void CausalLM::generate_worker_loop() {
    while (!should_terminate) {
        GenerationFuncArgs args;
        {
            std::unique_lock<std::mutex> lock(jobs_queue_mutex);
            new_job_condition.wait(lock, [this](){
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate)
                break;
            args = jobs.front();
            jobs.pop();
        }
        generate_impl(args.prompt, args.params, args.token_callback);
        args.done_callback();
    }
}

void CausalLM::stop_generation() {
    should_stop = true;
    return;
}