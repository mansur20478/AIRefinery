package com.final_project.airefinery;

import androidx.annotation.NonNull;

public enum EnhancementMode {
    FORMALIZE,
    PERSUADE,
    SIMPLIFY;

    @NonNull
    @Override
    public String toString() {
        switch (this) {
            case FORMALIZE:
                return "Formalize";
            case PERSUADE:
                return "Persuade";
            case SIMPLIFY:
                return "Simplify";
            default:
                return super.toString();
        }
    }

    public String toPrompt(String textToRefine, String extraContext) {
        String instruction = "No instruction";
        switch (this) {
            case FORMALIZE:
                instruction = "Make the text more formal without introducing any new information.";
                break;
            case PERSUADE:
                instruction = "Make the text sound more confident and persuasive, without exaggeration or adding information.";
                break;
            case SIMPLIFY:
                instruction = "Rewrite the text in simpler words, without adding anything new.";
                break;
        }
        if (!extraContext.isBlank()) {
            return "Context: " + extraContext + "\n"
                    + "Task: " + instruction + "\n"
                    + "Text: " + textToRefine + "\n"
                    + "Rewrite: ";
        }
        return "Task: " + instruction + "\n"
                + "Text: " + textToRefine + "\n"
                + "Rewrite: ";
    }
}