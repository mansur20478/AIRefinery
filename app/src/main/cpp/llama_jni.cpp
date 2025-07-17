#include <jni.h>
#include <string>
#include "causal_lm.h"

CausalLMConfig parse_causal_lm_config(JNIEnv* env, jobject jConfig) {
    CausalLMConfig config;

    jclass cls = env->GetObjectClass(jConfig);

    jfieldID fidModelPath = env->GetFieldID(cls, "modelPath", "Ljava/lang/String;");
    jstring jModelPath = (jstring)env->GetObjectField(jConfig, fidModelPath);
    const char* cModelPath = env->GetStringUTFChars(jModelPath, nullptr);
    config.modelPath = cModelPath;
    env->ReleaseStringUTFChars(jModelPath, cModelPath);
    return config;
}

GenerationParams parse_generation_params(JNIEnv* env, jobject jParams) {
    GenerationParams params;

    jclass cls = env->GetObjectClass(jParams);

    jfieldID fidMaxTokens = env->GetFieldID(cls, "maxTokens", "I");
    jfieldID fidTopK      = env->GetFieldID(cls, "topK", "I");
    jfieldID fidTopP      = env->GetFieldID(cls, "topP", "F");
    jfieldID fidTemp      = env->GetFieldID(cls, "temperature", "F");
    jfieldID fidGreedy    = env->GetFieldID(cls, "useGreedy", "Z");
    jfieldID fidThreads   = env->GetFieldID(cls, "n_threads", "I");

    params.maxTokens   = env->GetIntField(jParams, fidMaxTokens);
    params.topK        = env->GetIntField(jParams, fidTopK);
    params.topP        = env->GetFloatField(jParams, fidTopP);
    params.temperature = env->GetFloatField(jParams, fidTemp);
    params.useGreedy   = env->GetBooleanField(jParams, fidGreedy) == JNI_TRUE;
    params.n_threads   = env->GetIntField(jParams, fidThreads);

    return params;
}

CausalLM *g_lm = nullptr;
JavaVM *g_vm = nullptr;

extern "C" {

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *vm, void *reserved) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT jboolean JNICALL
Java_com_final_1project_airefinery_CausalLM_init(JNIEnv* env, jobject, jobject jConfig) {
    if (!g_lm) {
        g_lm = new CausalLM();
    }
    CausalLMConfig config = parse_causal_lm_config(env, jConfig);
    return g_lm->init(config) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_final_1project_airefinery_CausalLM_generate(JNIEnv* env, jobject, jstring jPrompt,
                                                      jobject jParams, jobject jCallback) {
    if (!g_lm) {
        LOG_ERROR("Model not initialized");
        return JNI_FALSE;
    }
    const char* promptChars = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(promptChars);
    env->ReleaseStringUTFChars(jPrompt, promptChars);

    GenerationParams params = parse_generation_params(env, jParams);

    jobject jGlobalCallback = env->NewGlobalRef(jCallback);

    TokenCallback tokenCallback = [jGlobalCallback](const std::string& token) -> bool {
        JNIEnv* env = nullptr;
        jint attachResult = g_vm->AttachCurrentThread(&env, nullptr);
        if (attachResult != JNI_OK || !env) {
            return false;
        }
        jclass callbackCls = env->GetObjectClass(jGlobalCallback);
        jmethodID onTokenMethod = env->GetMethodID(callbackCls, "onToken", "(Ljava/lang/String;)V");
        
        jstring jToken = env->NewStringUTF(token.c_str());
        env->CallVoidMethod(jGlobalCallback, onTokenMethod, jToken);
        
        env->DeleteLocalRef(jToken);
        env->DeleteLocalRef(callbackCls);
        g_vm->DetachCurrentThread();
        return true;
    };

    DoneCallback doneCallback = [jGlobalCallback]() -> void {
        JNIEnv* env = nullptr;
        jint attachResult = g_vm->AttachCurrentThread(&env, nullptr);
        if (attachResult != JNI_OK || !env) {
            return;
        }
        jclass callbackCls = env->GetObjectClass(jGlobalCallback);
        jmethodID onCompleteMethod = env->GetMethodID(callbackCls, "onComplete", "()V");

        env->CallVoidMethod(jGlobalCallback, onCompleteMethod);

        env->DeleteLocalRef(callbackCls);
        g_vm->DetachCurrentThread();
    };

    bool success = g_lm->generate(prompt.c_str(), params, tokenCallback, doneCallback);
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_final_1project_airefinery_CausalLM_stopGeneration(JNIEnv* env, jobject) {
    if (g_lm) {
        g_lm->stop_generation();
    }
}

JNIEXPORT void JNICALL
Java_com_final_1project_airefinery_CausalLM_free(JNIEnv*, jobject) {
    delete g_lm;
    g_lm = nullptr;
}

}