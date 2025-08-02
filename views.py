# views.py
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pickle
import os
from django.conf import settings

def UserHome(request):
    return render(request, 'users/userhome.html')

def training(request):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Embedding
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from django.shortcuts import render

    dataset = pd.read_csv(r'C:\Users\Admin\Desktop\CODE\Phishing_Url\media\malicious_balanced_dataset.csv')

    urls = dataset['url']
    labels = dataset['status']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(urls)

    X = tokenizer.texts_to_sequences(urls)
    max_length = 200
    X = pad_sequences(X, maxlen=max_length, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_length))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)

    model.save('malicious_url_cnn_model.h5')

    context = {
        'accuracy': f'{accuracy * 100:.2f}',
        'loss': f'{loss:.4f}',
    }
    return render(request, 'users/training.html', context)

def prediction(request):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    from django.shortcuts import render

    # Load the trained model
    model = load_model(r'C:\Users\Admin\Desktop\CODE\Phishing_Url\malicious_url_cnn_model.h5')

    if request.method == 'POST':
        url = request.POST.get('url')

        # Tokenizer settings used in training
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts([url])
        max_length = 200

        # Preprocess the URL
        X = tokenizer.texts_to_sequences([url])
        X = pad_sequences(X, maxlen=max_length, padding='post')

        # Make the prediction
        prediction = model.predict(X)
        prediction = 1 if prediction >= 0.5 else 0

        # Map prediction to label
        prediction_label = 'Malicious' if prediction == 1 else 'Safe'

        context = {
            'url': url,
            'prediction': prediction_label,
        }

        return render(request, 'users/prediction.html', context)

    return render(request, 'users/prediction.html')