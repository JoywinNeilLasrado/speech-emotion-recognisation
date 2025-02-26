<h2>Overview</h2>
<p>This project is a web-based application that utilizes a pre-trained deep learning model to recognize emotions from speech. The app processes an uploaded audio file, extracts relevant features, and predicts the speaker's emotion using a neural network.</p>

<h2>Features</h2>
<ul>
    <li>Upload an audio file in <code>.wav</code> format.</li>
    <li>Automatic feature extraction using MFCCs.</li>
    <li>Emotion classification into one of the following categories:</li>
    <ul>
        <li>Neutral</li>
        <li>Calm</li>
        <li>Happy</li>
        <li>Sad</li>
        <li>Angry</li>
        <li>Fearful</li>
        <li>Disgust</li>
        <li>Surprised</li>
    </ul>
    <li>User-friendly Gradio web interface.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>TensorFlow / Keras (Deep Learning Model)</li>
    <li>Librosa (Audio Processing)</li>
    <li>NumPy (Data Processing)</li>
    <li>Gradio (Web Interface)</li>
</ul>

<h2>Installation</h2>
<h3>Prerequisites</h3>
<p>Ensure you have Python 3.x installed along with the necessary dependencies.</p>

<h3>Install Required Libraries</h3>
<pre><code>pip install tensorflow librosa numpy gradio</code></pre>

<h2>Running the Application</h2>
<ol>
    <li>Train the model using <code>ml.py</code> if you wish to generate a new model:</li>
    <pre><code>python ml.py</code></pre>
    <li>Clone the repository or download the project files.</li>
    <li>Place the trained model file <code>emotion_recognition_model.h5</code> or <code>emotion_recognition_model.keras</code> in the project directory.</li>
    <li>Run the application:</li>
    <pre><code>python app.py</code></pre>
    <li>The Gradio web interface will launch, allowing you to upload an audio file and receive emotion predictions.</li>
</ol>

<h2>How It Works</h2>
<ol> 

  <h2>Future Enhancements</h2>
<ul>
    <li>Support for additional audio formats.</li>
    <li>Real-time emotion prediction.</li>
    <li>Improved model accuracy with a larger dataset.</li>
</ul>

<h2>License</h2>
<p>This project is open-source. Feel free to modify and distribute it as needed.</p>

<h2>Acknowledgments</h2>
<p>Inspired by research in speech emotion recognition.</p>
<p>Uses open-source libraries for audio processing and deep learning.</p> 
    <li>The user uploads an audio file.</li>
    <li>The file is processed using <code>librosa</code> to extract MFCC features.</li>
    <li>The features are normalized and reshaped for the deep learning model.</li>
    <li>The trained TensorFlow model predicts the most likely emotion.</li>
    <li>The predicted emotion is displayed in the web interface.</li>
</ol>

