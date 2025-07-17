package com.final_project.airefinery;

import java.lang.String;

public class CausalLM {
    static {
        System.loadLibrary("airefinery");
    }

    public CausalLM() {}

    public native boolean init(CausalLMConfig config);

    public native void generate(String prompt, GenerationParams params, TokenCallback callback);

    public native void stopGeneration();

    public native void free();
}
