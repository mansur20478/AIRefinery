package com.final_project.airefinery;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    public enum InferenceState {
        INITIALIZING,
        RUNNING,
        STOPPED,
        IDLE
    }
    public InferenceState currentState;

    public EditText extraContextEditText;
    public EditText inputTextEditText;
    public Spinner modeSpinner;
    public Button generateButton;
    public TextView outputTextView;
    public Button stopButton;
    public Button copyButton;
    public Button outputToInputButton;

    public ExaoneCausalLM clm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initUI();
        initModel();
    }

    public void initUI() {
        initInputOutputFields();
        initButtons();
    }

    public void initInputOutputFields() {
        inputTextEditText = findViewById(R.id.inputText);
        extraContextEditText = findViewById(R.id.extraContext);
        outputTextView = findViewById(R.id.outputText);
        modeSpinner = findViewById(R.id.modeSpinner);
        ArrayAdapter<EnhancementMode> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                EnhancementMode.values()
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner.setAdapter(adapter);
    }

    public void initButtons() {
        generateButton = findViewById(R.id.generateButton);
        generateButton.setOnClickListener(v -> {
            generateButtonClickHandler();
        });
        stopButton = findViewById(R.id.stopButton);
        stopButton.setOnClickListener(v -> {
            stopButtonClickHandler();
        });
        copyButton = findViewById(R.id.copyButton);
        copyButton.setOnClickListener(v -> {
            copyButtonClickHandler();
        });
        outputToInputButton = findViewById(R.id.outputToInputButton);
        outputToInputButton.setOnClickListener(v -> {
            outputToInputButtonClickHandler();
        });
    }

    public void initModel() {
        currentState = InferenceState.INITIALIZING;
        clm = new ExaoneCausalLM();
        showToast("Initializing");
        if (!clm.init(this)) {
            Log.e(TAG, "Failed to load model!");
            return;
        }
        Log.i(TAG, "Model loaded successfully");
        switchToIdleState();
    }

    public void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    public void switchToRunningState() {
        showToast("Running");
        currentState = InferenceState.RUNNING;

        stopButton.setVisibility(View.VISIBLE);
        generateButton.setVisibility(View.GONE);
        copyButton.setVisibility(View.GONE);
        outputToInputButton.setVisibility(View.GONE);

        inputTextEditText.setEnabled(false);
        extraContextEditText.setEnabled(false);
        modeSpinner.setEnabled(false);
    }

    public void switchToIdleState() {
        showToast("Done");
        currentState = InferenceState.IDLE;
        generateButton.setVisibility(View.VISIBLE);
        if (!outputTextView.getText().toString().isBlank()
          && !outputTextView.getText().toString().equals(getString(R.string.output_text))) {
            copyButton.setVisibility(View.VISIBLE);
            outputToInputButton.setVisibility(View.VISIBLE);
        }
        else {
            copyButton.setVisibility(View.GONE);
            outputToInputButton.setVisibility(View.GONE);
        }
        stopButton.setVisibility(View.GONE);

        inputTextEditText.setEnabled(true);
        extraContextEditText.setEnabled(true);
        modeSpinner.setEnabled(true);
    }

    public void switchToStoppedState() {
        showToast("Stopping");
        currentState = InferenceState.STOPPED;
        stopButton.setVisibility(View.VISIBLE);
        generateButton.setVisibility(View.GONE);
        copyButton.setVisibility(View.GONE);
        outputToInputButton.setVisibility(View.GONE);
    }

    public String constructPrompt() {
        int position = modeSpinner.getSelectedItemPosition();
        EnhancementMode selectedMode = EnhancementMode.values()[position];
        String textToRefine = inputTextEditText.getText().toString();
        String extraContext = extraContextEditText.getText().toString();
        return selectedMode.toPrompt(textToRefine, extraContext);
    }

    public void generateButtonClickHandler() {
        if (currentState != InferenceState.IDLE)
            return;
        String prompt = constructPrompt();
        if (prompt.isBlank())
            return;

        GenerationParams params = new GenerationParams();
        StringBuilder generatedText = new StringBuilder();
        TokenCallback callback = new TokenCallback() {
            @Override
            public void onToken(String token) {
                generatedText.append(token);
                runOnUiThread(() -> outputTextView.setText(generatedText.toString()));
            }

            @Override
            public void onComplete() {
                runOnUiThread(() -> switchToIdleState());
            }
        };

        clm.generate(prompt, params, callback);
        switchToRunningState();
    }

    public void stopButtonClickHandler() {
        if (currentState != InferenceState.RUNNING)
            return;
        clm.stopGeneration();
        switchToStoppedState();
    }

    public void copyButtonClickHandler() {
        String refinedText = outputTextView.getText().toString();

        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        ClipData clip = ClipData.newPlainText("Refined Text", refinedText);
        clipboard.setPrimaryClip(clip);

        showToast("Copied to clipboard");
    }

    public void outputToInputButtonClickHandler() {
        String refinedText = outputTextView.getText().toString();
        inputTextEditText.setText(refinedText);
    }
}
