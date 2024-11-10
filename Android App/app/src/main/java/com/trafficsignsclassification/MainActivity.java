package com.trafficsignsclassification;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.net.Uri;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.text.style.SuperscriptSpan;
import android.util.Log;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;
import android.view.WindowManager;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST_CODE = 101;
    private TextureView textureView;
    private TextView classTextView, probabilityTextView;
    private CameraDevice cameraDevice;
    private CameraManager cameraManager;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    private Interpreter tflite;
    private Handler frameHandler;
    private Runnable frameRunnable;
    private List<String> labels;
    private boolean isFlashOn = false;
    private String cameraId;
    private ImageButton flashToggleButton;
    private static final float THRESHOLD = 0.7f; // Probability threshold


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button openSettingsButton = findViewById(R.id.openSettingsButton);
        openSettingsButton.setOnClickListener(v -> {
            // Open the app settings
            Intent intent = new Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.setData(Uri.parse("package:" + getPackageName()));
            startActivity(intent);
        });

        // Find the main elements to animate
        View appLogo = findViewById(R.id.appLogo);

        // Load the fade-in and scale-up animation
        Animation fadeInScaleUp = AnimationUtils.loadAnimation(this, R.anim.fade_in_scale_up);

        // Start animations with a delay for a staggered effect
        appLogo.startAnimation(fadeInScaleUp);


        // Initialize the flash toggle button
        flashToggleButton = findViewById(R.id.flashToggleButton);

        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        try {
            // Get the camera ID for the back-facing camera
            for (String id : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(id);
                if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) {
                    cameraId = id;
                    break;
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

        // Set up the flash toggle button click listener
        flashToggleButton.setOnClickListener(v -> toggleFlashlight());

        // Keep the screen on while the camera is active
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Find the TextView for the project name
        TextView projectNameTextView = findViewById(R.id.projectNameTextView);

        // Create the text with "BETA" in superscript
        String text = "Traffic Signs Classification BETA";
        SpannableString spannableString = new SpannableString(text);

        // Set the "BETA" part to be superscript and smaller in size
        int betaStartIndex = text.indexOf("BETA");
        spannableString.setSpan(new SuperscriptSpan(), betaStartIndex, text.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        spannableString.setSpan(new RelativeSizeSpan(0.5f), betaStartIndex, text.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        // Set the spannable string to the TextView
        projectNameTextView.setText(spannableString);

        // Set the custom status bar color
        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.customBlack));

        // Existing code for loading labels
        try {
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open("labels.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            labels = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
            reader.close();
            Log.d("MainActivity", "Labels loaded successfully: " + labels.size() + " labels.");
        } catch (IOException e) {
            Log.e("MainActivity", "Error loading labels manually: " + e.getMessage());
        }

        // OpenCV initialization
        if (OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "OpenCV loaded successfully", Toast.LENGTH_SHORT).show();
        } else {
//            Toast.makeText(this, "Failed to load OpenCV", Toast.LENGTH_SHORT).show();
        }

        // Initialize TextureView and TextViews
        textureView = findViewById(R.id.textureView);
        classTextView = findViewById(R.id.classTextView);
        probabilityTextView = findViewById(R.id.probabilityTextView);


        // Initialize the Handler and Runnable for periodic frame processing
        frameHandler = new Handler();
        frameRunnable = new Runnable() {
            @Override
            public void run() {
                if (textureView != null) {
                    Bitmap bitmap = textureView.getBitmap();
                    if (bitmap != null) {
                        processImage(bitmap); // Process the captured frame
                    }
                }
                frameHandler.postDelayed(this, 100); // Adjust interval (100 ms) as needed
            }
        };
        frameHandler.post(frameRunnable); // Start periodic frame processing

        // Ensure TextureView is visible
        if (textureView != null) {
            textureView.setVisibility(View.VISIBLE);
        } else {
            throw new RuntimeException("TextureView is not found!");
        }

        // Load the TFLite model and labels
        try {
            tflite = new Interpreter(loadModelFile());
            Log.d("MainActivity", "Model loaded successfully."); // Add this line
            labels = FileUtil.loadLabels(this, "labels.txt");
        } catch (IOException e) {
            Log.e("MainActivity", "Error loading model or labels: " + e.getMessage());
            e.printStackTrace();
        }

        // Start the background thread
        startBackgroundThread();

        // Request camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        } else {
            setupCamera();
        }
    }

    private void toggleFlashlight() {
        if (cameraManager == null || cameraId == null) {
            Log.e("MainActivity", "CameraManager or CameraId is null");
            Toast.makeText(this, "Flashlight not supported on this device", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // Check if the device has a flashlight
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
            Boolean hasFlash = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
            if (hasFlash == null || !hasFlash) {
                Log.e("MainActivity", "This device does not have a flashlight");
                Toast.makeText(this, "This device does not support a flashlight", Toast.LENGTH_SHORT).show();
                return;
            }
        } catch (CameraAccessException e) {
            Log.e("MainActivity", "CameraAccessException: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Error accessing camera features", Toast.LENGTH_SHORT).show();
            return;
        }

        // Check if the camera permission is granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.e("MainActivity", "Camera permission not granted");
            Toast.makeText(this, "Camera permission not granted", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // Temporarily release the camera before toggling the flashlight
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }

            isFlashOn = !isFlashOn; // Toggle the flash state
            cameraManager.setTorchMode(cameraId, isFlashOn); // Turn the flash on or off
            flashToggleButton.setImageResource(isFlashOn ? R.drawable.outline_flash_off_24 : R.drawable.outline_flash_on_24);
            Log.d("MainActivity", "Flashlight toggled successfully: " + (isFlashOn ? "ON" : "OFF"));

            // Reinitialize the camera after toggling the flashlight
            setupCamera();

        } catch (CameraAccessException e) {
            Log.e("MainActivity", "Error toggling flashlight: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Error toggling flashlight", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e("MainActivity", "Unexpected error toggling flashlight: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Unexpected error occurred", Toast.LENGTH_SHORT).show();
        }
    }

    private void setupCamera() {
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        if (textureView.isAvailable()) {
            Log.d("MainActivity", "TextureView is available, opening camera...");
            openCamera();
        } else {
            Log.d("MainActivity", "TextureView is not available, setting listener...");
            textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    Log.d("MainActivity", "SurfaceTexture is now available, opening camera...");
                    openCamera();
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                    return false;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
            });
        }
    }


    private MappedByteBuffer loadModelFile() throws IOException {
        // Use try-with-resources to ensure the file descriptor is closed
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd("model_trained.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }


    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        // Turn off the flashlight when the app is paused
        if (isFlashOn) {
            toggleFlashlight();
        }

        // Release camera resources when the activity is paused
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }

        // Stop the background thread
        stopBackgroundThread();

        // Remove frame processing callbacks
        if (frameHandler != null) {
            frameHandler.removeCallbacks(frameRunnable);
        }
    }


    @Override
    protected void onResume() {
        super.onResume();

        // Check the camera permission status when the app resumes
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // Permission is granted, show the camera views and set up the camera
            showCameraViews();
            setupCamera();
        } else {
            // Permission is not granted, hide the camera views
            hideCameraViews();
        }

        // Restart the background thread
        startBackgroundThread();

        // Reopen the camera and reinitialize resources if needed
        if (textureView != null && textureView.isAvailable()) {
            openCamera();
        } else if (textureView != null) {
            textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    openCamera();
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                    return false;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
            });
        }

        // Restart frame processing
        if (frameHandler != null && frameRunnable != null) {
            frameHandler.post(frameRunnable);
        }
    }

    @Override
    protected void onStop() {
        super.onStop();

        // Release additional resources if necessary
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, show the camera views
                showCameraViews();

                // Introduce a small delay before setting up the camera
                new Handler().postDelayed(this::setupCamera, 1000);
            } else {
                // Permission denied, hide the camera views
                hideCameraViews();
            }
        }
    }

    private void hideCameraViews() {
        textureView.setVisibility(View.GONE);
        flashToggleButton.setVisibility(View.GONE);
        classTextView.setVisibility(View.GONE);
        probabilityTextView.setVisibility(View.GONE);
        findViewById(R.id.instructionTextView).setVisibility(View.GONE);
        findViewById(R.id.infoIcon).setVisibility(View.GONE);
        findViewById(R.id.permissionLayout).setVisibility(View.VISIBLE);
    }

    private void showCameraViews() {
        textureView.setVisibility(View.VISIBLE);
        flashToggleButton.setVisibility(View.VISIBLE);
        classTextView.setVisibility(View.VISIBLE);
        probabilityTextView.setVisibility(View.VISIBLE);
        findViewById(R.id.instructionTextView).setVisibility(View.VISIBLE);
        findViewById(R.id.infoIcon).setVisibility(View.VISIBLE);
        findViewById(R.id.permissionLayout).setVisibility(View.GONE);
    }

    private void openCamera() {
        try {
            if (backgroundHandler == null) {
                startBackgroundThread(); // Ensure the background thread is started
            }
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            cameraManager.openCamera(cameraManager.getCameraIdList()[0], new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    try {
                        cameraDevice = camera;
                        createCameraPreviewSession();
                    } catch (Exception e) {
                        Log.e("MainActivity", "Error while creating preview session: " + e.getMessage());
                        // Retry opening the camera
                        openCamera();
                    }
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    Log.d("MainActivity", "Camera disconnected.");
                    camera.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    Log.e("MainActivity", "Error opening camera: " + error);
                    camera.close();
                    cameraDevice = null;
                }
            }, backgroundHandler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(textureView.getWidth(), textureView.getHeight());
            Surface surface = new Surface(texture);

            CaptureRequest.Builder captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            cameraDevice.createCaptureSession(List.of(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (cameraDevice == null) {
                        return;
                    }
                    try {
                        session.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
//                    Toast.makeText(MainActivity.this, "Camera setup failed", Toast.LENGTH_SHORT).show();
                }
            }, backgroundHandler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void processImage(Bitmap bitmap) {
        if (bitmap == null) {
            Log.e("MainActivity", "Bitmap is null!");
            return;
        }

        // Convert Bitmap to OpenCV Mat
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        // Apply preprocessing: convert to grayscale
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);

        // Apply histogram equalization
        Imgproc.equalizeHist(mat, mat);

        // Resize the image to 32x32
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new org.opencv.core.Size(32, 32));

        // Normalize the pixel values to be between 0 and 1
        resizedMat.convertTo(resizedMat, CvType.CV_32F, 1.0 / 255); // Convert to CV_32F and scale

        // Prepare input for the TFLite model
        float[][][][] input = new float[1][32][32][1];
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                // Extract the normalized pixel value
                input[0][i][j][0] = (float) resizedMat.get(i, j)[0];
            }
        }

        // Run inference and log output
        float[][] output = new float[1][labels.size()];
        try {
            tflite.run(input, output);
        } catch (Exception e) {
            Log.e("MainActivity", "Error running TFLite model: " + e.getMessage());
            return;
        }

        // Log the output values for debugging
        for (int i = 0; i < output[0].length; i++) {
            Log.d("MainActivity", "Class " + i + ": " + output[0][i]);
        }

        // Find the class with the highest probability
        int classIndex = getMaxIndex(output[0]);
        float probability = output[0][classIndex];

        // Debug: Log selected class and probability
        Log.d("MainActivity", "Selected Class Index: " + classIndex);
        Log.d("MainActivity", "Probability: " + probability);

        // Update the UI on the main thread
        runOnUiThread(() -> {
            int customBlack = ContextCompat.getColor(this, R.color.customBlack);
            int customGreen = ContextCompat.getColor(this, R.color.deep_green);

            if (probability > THRESHOLD) {
                // Valid class name
                SpannableString classText = new SpannableString("Class: " + labels.get(classIndex));
                classText.setSpan(new ForegroundColorSpan(customGreen), 6, classText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                classTextView.setText(classText);
                classTextView.setTextColor(customBlack); // "Class: " portion remains custom white

                // Probability text with green color
                SpannableString probabilityText = new SpannableString("Probability: " + String.format("%.2f%%", probability * 100));
                probabilityText.setSpan(new ForegroundColorSpan(customGreen), 12, probabilityText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                probabilityTextView.setText(probabilityText);
                probabilityTextView.setTextColor(customBlack); // "Probability: " portion remains custom white
            } else {
                // "N/A" class name
                SpannableString classText = new SpannableString("Class: N/A");
                classText.setSpan(new ForegroundColorSpan(Color.RED), 6, classText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                classTextView.setText(classText);
                classTextView.setTextColor(customBlack); // "Class: " portion remains custom white

                // Probability text with red color
                SpannableString probabilityText = new SpannableString("Probability: N/A");
                probabilityText.setSpan(new ForegroundColorSpan(Color.RED), 12, probabilityText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                probabilityTextView.setText(probabilityText);
                probabilityTextView.setTextColor(customBlack); // "Probability: " portion remains custom white
            }
        });
    }


    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}