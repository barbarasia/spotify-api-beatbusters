import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st

CLIENT_ID = os.getenv("spotipy_id")
CLIENT_SECRET = os.getenv("spotipy_secret")

#Initialize SpotiPy with user credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                           client_secret=CLIENT_SECRET))
df = pd.read_csv('/Users/Barbara/Desktop/Ironhack/Labs/Week_6/spotipy-api/tracks_clustered_df.csv')
df_tf = pd.read_csv('/Users/Barbara/Desktop/Ironhack/Labs/Week_6/spotipy-api/tracks_and_features.csv')


def check_song_in_dataset(track_id, dataframe):
    return track_id in dataframe['track_id'].values

st.title("Recommend a song!")


fav_song = st.text_input("What's your favorite song?", key="fav_song")
fav_song_artist = st.text_input("Who's the artist?", key="fav_song_artist")
if st.button('Submit'):
    # Search for the song on Spotify
    results = sp.search(q="track:" + fav_song + " artist:" + fav_song_artist, limit=1)

    # Check if a track was found
    if results['tracks']['items']:
        # Extract the track details
        track = results['tracks']['items'][0]
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        track_id = track['id']

        # Display the track details
        st.write(f"Found track: {track_name} by {track_artist}")

        # Display the Spotify embed
        st.components.v1.iframe(f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator",
                                width=320, height=80, scrolling=False)

        # Ask for confirmation
        response = st.selectbox('Is it the song?', ['Yes', 'No'], key='confirmation')
        
        if response == 'Yes':
            # Check if the track ID is in the dataset
            is_in_dataset = check_song_in_dataset(track_id, df)

            if is_in_dataset:
                # Recommend a song directly if the song is in the dataset
                cluster_number = df[df['track_id'] == track_id]['cluster_km100'].iloc[0]
                recommendation = df[df['cluster_km100'] == cluster_number].sample(1)
                st.write(f"Recommended track: {recommendation['track_name'].iloc[0]} by {recommendation['artist_name'].iloc[0]}")
                st.components.v1.iframe(f"https://open.spotify.com/embed/track/{recommendation['track_id'].iloc[0]}?utm_source=generator",
                                        width=320, height=80, scrolling=False)
            else:
                # Extract audio features and add the song to the dataset if not already in it
                list_of_audio_features_fav = sp.audio_features(track_id)[0]

                audio_features_dict = {
                    'danceability': [list_of_audio_features_fav['danceability']],
                    'energy': [list_of_audio_features_fav['energy']],
                    'key': [list_of_audio_features_fav.get('key', -1)],
                    'loudness': [list_of_audio_features_fav['loudness']],
                    'mode': [list_of_audio_features_fav['mode']],
                    'speechiness': [list_of_audio_features_fav['speechiness']],
                    'acousticness': [list_of_audio_features_fav['acousticness']],
                    'instrumentalness': [list_of_audio_features_fav['instrumentalness']],
                    'liveness': [list_of_audio_features_fav['liveness']],
                    'valence': [list_of_audio_features_fav['valence']],
                    'tempo': [list_of_audio_features_fav['tempo']],
                    'duration_ms': [list_of_audio_features_fav['duration_ms']],
                    'time_signature': [list_of_audio_features_fav['time_signature']],
                    'track_href': [list_of_audio_features_fav['track_href']]
                }

                audio_features_df = pd.DataFrame(audio_features_dict)

                artist_ids = [track['artists'][0]['id']]
                artist_names = [track['artists'][0]['name']]
                track_ids = [track['id']]
                track_href_1 = [track['href']]
                track_names = [track['name']]
                popularity = [track['popularity']]
                is_explicit = [track['explicit']]
                durations_ms = [track['duration_ms']]
                album_release_dates = [track['album']['release_date']]
                album_release_date_precisions = [track['album']['release_date_precision']]

                tracks_df_fav = pd.DataFrame({
                    'artist_id': artist_ids,
                    'artist_name': artist_names,
                    'track_id': track_ids,
                    'track_name': track_names,
                    'album_release_date': album_release_dates,
                    'album_release_date_precision': album_release_date_precisions,
                    'is_explicit': is_explicit,
                    'durations_ms': durations_ms,
                    'popularity': popularity,
                    'track_href_1': track_href_1
                })

                merged_fav = pd.merge(tracks_df_fav, audio_features_df, left_on='track_href_1', right_on='track_href')
                merged_fav = merged_fav.drop(columns=['track_href_1'])

                # Update the dataset with the new track
                df = pd.concat([df, merged_fav], axis=0, ignore_index=True)

                # Recompute the clustering with the new data
                audio_features_model_on = ['popularity', 'danceability', 'energy', 'key',
                                           'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                                           'liveness', 'valence', 'tempo']
                features_df = df[audio_features_model_on]

                km300 = KMeans(300)
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features_df)
                km300.fit(scaled_features)

                df['cluster_km100'] = km300.labels_

                # Get the cluster number for the new track
                cluster_number = df[df['track_id'] == track_id]['cluster_km100'].iloc[0]

                # Recommend a song from the same cluster
                recommendation = df[df['cluster_km100'] == cluster_number].sample(1)
                st.write(f"Recommended track: {recommendation['track_name'].iloc[0]} by {recommendation['artist_name'].iloc[0]}")
                st.components.v1.iframe(f"https://open.spotify.com/embed/track/{recommendation['track_id'].iloc[0]}?utm_source=generator",
                                        width=320, height=80, scrolling=False)
    
        else:
            st.write('Are you sure the song exists? Try again.')
    else:
        st.write('No matching song found. Please check the details and try again.')
