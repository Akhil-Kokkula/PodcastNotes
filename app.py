from flask import Flask, render_template, request, redirect, url_for
import requests
from listennotes import podcast_api
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import pyaudioconvert as pac
import soundfile as sf
import librosa as lb
import wave
import transformers
from transformers import pipeline # type: ignore
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction
from azure.core.credentials import AzureKeyCredential
#import assemblyai as aai

app = Flask(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SERVICE_REGION = os.getenv("SERVICE_REGION")
LANGUAGE_KEY = os.getenv("LANGUAGE_KEY")
ENDPOINT = os.getenv("ENDPOINT")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return redirect(url_for("search", episode_name=request.form["episode_name"]))
    return render_template("home.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        return redirect(url_for("search", episode_name=request.form["episode_name"]))
    episode_name = request.args.get("episode_name")
    # client = podcast_api.Client(api_key=None)
    # response = client.search(q='star wars', type='episode')
    client = podcast_api.Client(api_key=API_KEY)
    response = client.search(q=episode_name, type='episode')
    print(episode_name)
    data = json.loads(response.text)
    episode_results = data["results"]
    return render_template("episode_results_page.html", episode_results=episode_results)

# call the api to transcript the episode and feed to Azure Cognitive Service Language to generate notes
@app.route("/notes", methods=["GET", "POST"])
def notes_page():
    episode_audio = str(request.args.get("episode_audio"))
    episode_id = request.args.get("episode_id")
    episode_name = "episode-" + str(episode_id) + ".wav"
    r = requests.get(episode_audio)
    with open(episode_name, "wb") as f:
        f.write(r.content)
    data, samplerate = sf.read(episode_name)
    episode_name_formatted = "episode-formatted-" + str(episode_id) + ".wav"
    sf.write(episode_name_formatted, data, samplerate, subtype='PCM_16')
    os.remove(episode_name)

    speech_recognize_continuous_from_file(episode_name_formatted, episode_id)
    
    with open(str(episode_id)+".txt", "r") as f:
        transcript = f.read()
    print(transcript)
    transcript_arr = []
    # for piece in splitter(800, transcript):
    #     transcript_arr.append(piece)
    transcript_arr.append(transcript)

    client = authenticate_client()
    notes = sample_extractive_summarization(client, transcript_arr)
    
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # for i in transcript_arr:
    #     summary = summarizer(i, max_length=200, min_length=200, do_sample=False)
    #     notes += summary[0]['summary_text'] + "\n" # type: ignore
    # print(notes)

    return render_template("notes.html", notes=notes)


def splitter(n, s):
    pieces = s.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def authenticate_client():
    ta_credential = AzureKeyCredential(str(LANGUAGE_KEY))
    text_analytics_client = TextAnalyticsClient(
            endpoint=str(ENDPOINT), 
            credential=ta_credential)
    return text_analytics_client


# Example method for summarizing text
def sample_extractive_summarization(client, document):
    print(document)
    poller = client.begin_analyze_actions(
        document,
        actions=[
            ExtractiveSummaryAction(max_sentence_count=20)
        ],
    )

    document_results = poller.result()
    for result in document_results:
        extract_summary_result = result[0]  # first document, first result
        if extract_summary_result.is_error:
            print("...Is an error with code '{}' and message '{}'".format(
                extract_summary_result.code, extract_summary_result.message
            ))
        else:
            return "Summary extracted: \n{}".format(
                " ".join([sentence.text for sentence in extract_summary_result.sentences]))


def speech_recognize_continuous_from_file(file, episode_id):
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=file)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    output_file = open(str(episode_id)+".txt", "w")

    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        output_file.close()
        nonlocal done
        done = True

    # all_results = []
    def handle_final_result(evt):
        output_file.write(evt.result.text)
        # all_results.append(evt.result.text)

    speech_recognizer.recognized.connect(handle_final_result)
    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    print("Printing all results:")
    # print(all_results)

    # df = pd.DataFrame(all_results)

    # file_name = TEXT_FILE_LOCATION + "episode-" + str(episode_id) + "-speech-to-text-csv-output.csv"
    # df.to_string(file_name)
    # df.to_csv(file_name)


    print ("Audio File: "+file+" converted successfully")
    print ("####################################################################################")

