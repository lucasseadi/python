from googleapiclient.discovery import build
from pydub import AudioSegment
from pytube import YouTube
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os


# Function to search YouTube for videos matching the query
def search_youtube(query):
    print(f"Searching for '{query}'...")
    # Set up the YouTube Data API client
    api_key = 'AIzaSyDEewuOEe1NvQh_NUC9OTc8MHCAscNMgwQ'  # Replace with your YouTube Data API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Search for videos
    video_urls = []
    next_page_token = None
    while True:
        request = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        # Extract video URLs
        for item in response['items']:
            video_id = item['id']['videoId']
            url = f'https://www.youtube.com/watch?v={video_id}'
            video_urls.append(url)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_urls


# Function to download audio in MP3 format from YouTube video
def download_audio(video_url, output_path):
    print(f"Downloading {video_url} to {output_path}...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode (without opening a window)
    driver = webdriver.Chrome(options=options)

    try:
        # downloads video as mp4
        driver.get(video_url)
        yt = YouTube(driver.current_url)
        audio = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        audio.download(output_path=output_path)
        file_path = audio.default_filename

        # changes its extension to mp3
        new_file_path = file_path[:-4] + '.mp3'
        audio_file = os.path.join(output_path, file_path)
        new_audio_file = os.path.join(output_path, new_file_path)
        os.rename(audio_file, new_audio_file)

        # converts it to wav
        new_audio_wav = new_audio_file[:-4] + '.wav'
        sound = AudioSegment.from_mp3(new_audio_file)
        sound.export(new_audio_wav, format="wav")
    finally:
        driver.quit()


if __name__ == "__main__":
    # Example usage
    search_query = "earnings call"
    output_path = "./audio_files/"

    video_urls = search_youtube(search_query)

    for video_url in video_urls:
        download_audio(video_url, output_path)
