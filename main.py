import pandas as pd
import time
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report



client_id = '585f899c23764e89a28b2af635df68fc'
client_secret = '0e5c3b3dd21949b79add3a4ee941cdb8'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)




def get_spotify_data_spotipy():
    
    data = pd.read_csv('playlists.csv')

    audio_features = ["duration_ms", "loudness", "tempo", "acousticness", "danceability", "liveness", "instrumentalness", "energy", "speechiness", "valence"]

    # print(data, "\n\n")

    start_index = 560
    
    # iterar as linhas do CSV
    for index in range(start_index, len(data)):
        avg_features = []
        row = data.iloc[index]  
        print("i = ", index)

        playlist_url = row['Playlist_link']
        
        # id vem a seguir a barra
        playlist_id = playlist_url.split('/')[-1]

        # obter tracks da playlist
        playlist_tracks = sp.playlist_tracks(playlist_id, limit=50)

        # obter numero de tracks da playlist
        playlist_size = playlist_tracks['total']
        
        # inicializar features da playlist
        feature_values = {feature: [] for feature in audio_features}
        
        count = 0
        for track in playlist_tracks['items']:

            # casos em que nao sao tracks mas sim episodios de podcasts
            # ou entao tracks indisponiveis
            if track['track'] == None or track['track']['id'] == None:
                if playlist_size < 100:
                    if count == playlist_size:  # ultimo, break para guardar
                        break
                    else:
                        count += 1
                        continue    # nao e o ultimo, continuar
                else:   # playlist tem mais que 100, analisar os primeiros 100 (0-99)
                    if count == 99: # ultimo da amostra
                        break
                    else:
                        count += 1
                        continue

            # id de cada cancao
            track_id = track['track']['id']

            # nome da cancao
            # print(track['track']['name'], "\n\n")            
            
            # handle do erro 429 - too many requests
            try:
                # extrair todas as features da cancao
                audio_features_data = sp.audio_features(track_id)
            except SpotifyException as e:
                if e.http_status == 429:
                    print(f"Received a 429 error for track {track['track']['name']}.\nError message:\n")
                    print(e.msg, "\nExiting...\n")
                    exit(-1)
                    
            count += 1

            # print(audio_features_data, "\n")
            # print(audio_features, "\n\n")

            # adicionar as features que queremos (lista audio_features)
            if audio_features_data is not None and audio_features_data[0] is not None:
                for feature in audio_features:
                    feature_values[feature].append(audio_features_data[0][feature])

            
        
        # calcular media de cada feature para a playlist
        playlist_avg_features = [sum(feature_values[feature]) / len(feature_values[feature]) for feature in audio_features]
        
        # adicionar MBTI type e medias
        avg_features.append([row['Type']] + playlist_avg_features)

        # criar data frame 
        feature_df = pd.DataFrame(avg_features, columns=['Type'] + audio_features)

        # guardar no csv
        feature_df.to_csv('playlist_features.csv', mode='a', header=False, index=False)

        count = 0





def model():

    data = pd.read_csv('playlist_features.csv')

    X = data.drop('Type', axis=1)
    y = data['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=4)  
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
    
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=10)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=6)

    svc_classifier = SVC(C=1,kernel='poly')

        
    classifiers = [rf_classifier, dt_classifier, knn_classifier, svc_classifier]

    avg_acc = 0.0
    avg_f1 = 0.0

    for i in range(0, len(classifiers)):
        acc, f1 = print_model_output(classifiers[i], X_train_pca, X_test_pca, y_train, y_test)
        avg_acc += acc
        avg_f1 += f1
    
    avg_acc /= len(classifiers)
    avg_f1 /= len(classifiers)
    print("\n --- AVERAGE ACCURACY ---\n" + str (avg_acc) + "\n\n")
    print("\n --- AVERAGE F1 SCORE ---\n" + str (avg_f1))
        

    


def print_model_output(classifier, X_train_pca, X_test_pca, y_train, y_test):
    print("Starting ", classifier)
    classifier.fit(X_train_pca, y_train)
    predictions = classifier.predict(X_test_pca)

    correct = accuracy_score(y_test, predictions)
    off_by_one = custom_accuracy(y_test, predictions)
    total_correct = correct + off_by_one

    precision, recall, f1_score = custom_f1_score(y_test, predictions)

    print("Accuracy:", correct)
    print("Accuracy when missing one letter:", off_by_one)
    print("Custom Accuracy (correct + off by one letter):", total_correct)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Custom F1 Score:", f1_score, "\n\n")
    

    return total_correct, f1_score





def custom_accuracy(y_true, y_pred):
    
    def is_off_by_one_letter(true_label, pred_label):
        if len(true_label) != len(pred_label):
            return False

        diff_count = sum(1 for t, p in zip(true_label, pred_label) if t != p)
        return diff_count == 1

    correct_count = sum(1 for true_label, pred_label in zip(y_true, y_pred) if is_off_by_one_letter(true_label, pred_label))
    
    return correct_count / len(y_true)  





def custom_f1_score(y_true, y_pred):
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label or are_labels_off_by_one(true_label, pred_label):
            true_positives += 1
        else:
            if pred_label != true_label:
                false_positives += 1
            else:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score





def are_labels_off_by_one(label1, label2):
    return len(label1) == len(label2) and sum(c1 != c2 for c1, c2 in zip(label1, label2)) == 1





def main():

    # get_spotify_data_spotipy()
   
    model()




if __name__ == '__main__':
    main()

