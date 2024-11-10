package com.trafficsignsclassification;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityOptionsCompat;

public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        // Make the activity full-screen and hide the status bar
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_LAYOUT_STABLE |
                        View.SYSTEM_UI_FLAG_FULLSCREEN
        );

        setContentView(R.layout.activity_splash);

        // Set app version text and animation
        TextView versionTextView = findViewById(R.id.versionTextView);
        versionTextView.setText("Version : 1.0 (Beta)");
        Animation versionTextViewAnim = AnimationUtils.loadAnimation(this, R.anim.text_animation);
        versionTextView.startAnimation(versionTextViewAnim);

        // Apply zoom-in animation to the logo
        ImageView appLogo = findViewById(R.id.appLogo);
        Animation zoomInAnimation = AnimationUtils.loadAnimation(this, R.anim.zoom_in);
        appLogo.startAnimation(zoomInAnimation);

        // Apply bounce animation to the title text
        TextView titleTextView = findViewById(R.id.titleTextView);
        Animation textAnimation = AnimationUtils.loadAnimation(this, R.anim.bounce);
        titleTextView.startAnimation(textAnimation);

        new Handler().postDelayed(() -> {
            Intent intent = new Intent(SplashActivity.this, MainActivity.class);
            startActivity(intent);

            // Apply fade-in and scale-up animation on transition
            overridePendingTransition(R.anim.fade_in_scale_up, android.R.anim.fade_out);

            finish(); // Close the splash activity
        }, 3000); // 3000 ms delay (3 seconds)
    }
}