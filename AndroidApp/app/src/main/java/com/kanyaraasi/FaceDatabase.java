package com.kanyaraasi;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * FaceDatabase
 *
 * Manages persistent storage of enrolled face embeddings and performs
 * cosine similarity search to identify matching individuals.
 */
public class FaceDatabase {
    private static final String TAG = "FaceDatabase";
    private static final String DB_FILENAME = "face_database.json";

    // Default threshold for MobileFaceNet 192-D cosine similarity (0.65 - 0.75 is standard)
    public static final float DEFAULT_SIMILARITY_THRESHOLD = 0.70f;

    private final Context context;
    private final Map<String, List<float[]>> enrolledFaces = new HashMap<>();

    public FaceDatabase(Context context) {
        this.context = context;
        loadDatabase();
    }

    /**
     * Enrolls a new face embedding for a person.
     * Multiple embeddings can be stored per person for increased angle/lighting accuracy.
     *
     * @param name      Name of the person (e.g. "Dhruv")
     * @param embedding 192-dimensional L2-normalized embedding vector
     */
    public synchronized void enrollFace(String name, float[] embedding) {
        if (name == null || name.trim().isEmpty() || embedding == null) {
            return;
        }
        String cleanName = name.trim();
        List<float[]> embeddings = enrolledFaces.computeIfAbsent(cleanName, k -> new ArrayList<>());
        embeddings.add(embedding);

        saveDatabase();
        Log.d(TAG, "Enrolled face for: " + cleanName + " (Total samples: " + embeddings.size() + ")");
    }

    /**
     * Finds the closest matching enrolled person for a given face embedding.
     *
     * @param queryEmbedding 192-dimensional query embedding
     * @param threshold      Minimum cosine similarity required (e.g. 0.70)
     * @return MatchResult containing the best match name and similarity, or null if below threshold
     */
    public synchronized MatchResult findBestMatch(float[] queryEmbedding, float threshold) {
        if (queryEmbedding == null || enrolledFaces.isEmpty()) {
            return null;
        }

        String bestName = null;
        float maxSimilarity = -1.0f;

        for (Map.Entry<String, List<float[]>> entry : enrolledFaces.entrySet()) {
            String name = entry.getKey();
            for (float[] storedEmb : entry.getValue()) {
                float sim = computeCosineSimilarity(queryEmbedding, storedEmb);
                if (sim > maxSimilarity) {
                    maxSimilarity = sim;
                    bestName = name;
                }
            }
        }

        if (bestName != null && maxSimilarity >= threshold) {
            return new MatchResult(bestName, maxSimilarity);
        } else {
            return new MatchResult(null, maxSimilarity); // No confident match
        }
    }

    /**
     * Computes the Cosine Similarity between two L2-normalized vectors.
     * For normalized vectors, cosine similarity is simply their dot product.
     */
    public static float computeCosineSimilarity(float[] v1, float[] v2) {
        if (v1 == null || v2 == null || v1.length != v2.length) {
            return 0.0f;
        }
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 == 0.0f || norm2 == 0.0f) {
            return 0.0f;
        }
        return (float) (dot / (Math.sqrt(norm1) * Math.sqrt(norm2)));
    }

    /**
     * Returns a list of all enrolled names.
     */
    public synchronized List<String> getEnrolledNames() {
        return new ArrayList<>(enrolledFaces.keySet());
    }

    /**
     * Returns total number of registered individuals.
     */
    public synchronized int getEnrolledCount() {
        return enrolledFaces.size();
    }

    /**
     * Removes an enrolled person from the database.
     */
    public synchronized boolean removePerson(String name) {
        if (enrolledFaces.remove(name) != null) {
            saveDatabase();
            return true;
        }
        return false;
    }

    /**
     * Saves the current face database to private internal storage as JSON.
     */
    private synchronized void saveDatabase() {
        try {
            JSONObject root = new JSONObject();
            JSONArray peopleArray = new JSONArray();

            for (Map.Entry<String, List<float[]>> entry : enrolledFaces.entrySet()) {
                JSONObject personObj = new JSONObject();
                personObj.put("name", entry.getKey());

                JSONArray embeddingsArray = new JSONArray();
                for (float[] emb : entry.getValue()) {
                    JSONArray embArray = new JSONArray();
                    for (float f : emb) {
                        embArray.put((double) f);
                    }
                    embeddingsArray.put(embArray);
                }
                personObj.put("embeddings", embeddingsArray);
                peopleArray.put(personObj);
            }

            root.put("people", peopleArray);

            File file = new File(context.getFilesDir(), DB_FILENAME);
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
            Log.d(TAG, "Face database saved successfully (" + enrolledFaces.size() + " people)");
        } catch (Exception e) {
            Log.e(TAG, "Failed to save face database", e);
        }
    }

    /**
     * Loads enrolled face records from private internal storage.
     */
    private synchronized void loadDatabase() {
        enrolledFaces.clear();
        File file = new File(context.getFilesDir(), DB_FILENAME);
        if (!file.exists()) {
            Log.d(TAG, "No existing face database found at: " + file.getAbsolutePath());
            return;
        }

        try (FileInputStream fis = new FileInputStream(file)) {
            int size = fis.available();
            byte[] buffer = new byte[size];
            fis.read(buffer);
            String jsonStr = new String(buffer, StandardCharsets.UTF_8);

            JSONObject root = new JSONObject(jsonStr);
            JSONArray peopleArray = root.getJSONArray("people");

            for (int i = 0; i < peopleArray.length(); i++) {
                JSONObject personObj = peopleArray.getJSONObject(i);
                String name = personObj.getString("name");
                JSONArray embeddingsArray = personObj.getJSONArray("embeddings");

                List<float[]> list = new ArrayList<>();
                for (int j = 0; j < embeddingsArray.length(); j++) {
                    JSONArray embArray = embeddingsArray.getJSONArray(j);
                    float[] emb = new float[embArray.length()];
                    for (int k = 0; k < embArray.length(); k++) {
                        emb[k] = (float) embArray.getDouble(k);
                    }
                    list.add(emb);
                }
                enrolledFaces.put(name, list);
            }
            Log.d(TAG, "Loaded face database with " + enrolledFaces.size() + " people: " + enrolledFaces.keySet());
        } catch (Exception e) {
            Log.e(TAG, "Failed to load face database", e);
        }
    }

    /**
     * Result of a face comparison.
     */
    public static class MatchResult {
        public final String name;
        public final float similarity;
        public final boolean isMatch;

        public MatchResult(String name, float similarity) {
            this.name = name;
            this.similarity = similarity;
            this.isMatch = (name != null);
        }
    }
}
