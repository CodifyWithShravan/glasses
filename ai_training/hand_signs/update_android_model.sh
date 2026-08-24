#!/bin/bash

echo "🚀 Copying newly trained model to Android App..."

cp sign_classifier.tflite ../../AndroidApp/app/src/main/assets/sign_classifier.tflite
cp sign_labels.txt ../../AndroidApp/app/src/main/assets/sign_labels.txt
if [ -f sign_weights.json ]; then
    cp sign_weights.json ../../AndroidApp/app/src/main/assets/sign_weights.json
    echo "✅ Copied sign_weights.json"
fi

echo "✅ Copied sign_classifier.tflite"
echo "✅ Copied sign_labels.txt"
echo ""
echo "📱 Next step: Rebuild the Android App in Android Studio or using Gradle!"
