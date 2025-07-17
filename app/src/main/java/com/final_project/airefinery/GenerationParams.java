package com.final_project.airefinery;

public class GenerationParams {
    public int maxTokens = 1024;
    public int topK = 40;
    public float topP = 0.8f;
    public float temperature = 1.0f;
    public boolean useGreedy = true;
    public int n_threads = -1;

    public GenerationParams() {}

    public GenerationParams(int maxTokens, int topK, float topP, float temperature, boolean useGreedy) {
        this.maxTokens = maxTokens;
        this.topK = topK;
        this.topP = topP;
        this.temperature = temperature;
        this.useGreedy = useGreedy;
    }
}