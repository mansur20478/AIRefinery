package com.final_project.airefinery;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.content.Context;
import android.util.Log;

public class ExaoneCausalLM {
    private static final String TAG = "ExaoneCausalLM";
    private static final String MODEL_FILENAME = "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf";
    private CausalLM model;

    public ExaoneCausalLM() {
        model = new CausalLM();
    }

    private String applyTemplate(String prompt) {
        String systemStr = "<|system|>You are EXAONE model from LG AI Research, a helpful assistant.";
        String userStr = "<|user|>" + prompt;
        String assistantStr = "<|assistant|>";
        return systemStr + "\n" + userStr + "\n" + assistantStr;
    }

    public boolean init(Context context) {
        try {
            File outFile = new File(context.getFilesDir(), MODEL_FILENAME);
            if (!outFile.exists()) {
                try (InputStream in = context.getAssets().open(MODEL_FILENAME);
                     FileOutputStream out = new FileOutputStream(outFile)) {

                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                }
            }
            CausalLMConfig config = new CausalLMConfig();
            config.modelPath = outFile.getAbsolutePath();
            return model.init(config);
        } catch (Exception e) {
            Log.e(TAG, "Failed to init", e);
            return false;
        }
    }

    public void generate(String prompt, GenerationParams params, TokenCallback callback) {
        String newPrompt = applyTemplate(prompt);
        model.generate(newPrompt, params, callback);
    }

    public void stopGeneration() {
        model.stopGeneration();
    }

    public void free() {
        model.free();
    }
}
