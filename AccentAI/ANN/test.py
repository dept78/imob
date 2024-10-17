import subprocess
import discord
from discord.ext import commands
from discord import Member
from discord.ext.tasks import loop
import asyncio
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify credentials (replace with your own)
SPOTIPY_CLIENT_ID = "586b706aac0b4f2ebceef17aef9127c1"
SPOTIPY_CLIENT_SECRET = "6914d639ca9d441b92aeb0f8b70398e0"

# Initialize Spotify client
spotify_client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

intents = discord.Intents().all()
intents.presences = True
client = commands.Bot(command_prefix="st ", intents=intents)
client.remove_command('help')

account = None
prev_song = None

# Function to get Spotify track ID by song name
def get_track_id(song_name):
    results = spotify_client.search(q=song_name, type="track", limit=1)
    tracks = results.get('tracks', {}).get('items', [])
    if tracks:
        return tracks[0]['id'], tracks[0]['name']
    return None, None

# Function to play a song using track ID
def player(track_id):
    track_url = f"https://open.spotify.com/track/{track_id}"  # Spotify URL
    # Use VLC to play the track in the terminal
    subprocess.run(["vlc", "--play-and-exit", track_url])  # Add '--intf dummy' to run without GUI

@client.event
async def on_ready():
    print_info.start()
    print(f"im alive and working!! (logged in as {client.user})")
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="SUIIIIIIIIIII"))

@client.command()
async def join(ctx, user: Member):
    global account
    account = user
    await ctx.send(f"i will now stalk {account.display_name} :smiling_imp: :smiling_imp: ")

@loop(seconds=1)
async def print_info():
    global account, prev_song
    if account:
        try:
            if account.activities[0].track_id != prev_song:
                player(account.activities[0].track_id)
                print(f"Now playing {account.activities[0].title}")
                prev_song = account.activities[0].track_id
        except Exception:
            pass

# Function to take terminal input and execute commands
async def terminal_input():
    await client.wait_until_ready()
    while not client.is_closed():
        command = input("Enter a command (play song name, stop, etc.): ").strip().lower()
       
        if command.startswith("play"):
            try:
                # Extract song name
                song_name = " ".join(command.split(" ")[1:])
                track_id, track_name = get_track_id(song_name)
                if track_id:
                    player(track_id)
                    print(f"Playing: {track_name} (ID: {track_id})")
                else:
                    print("Song not found. Try another name.")
            except IndexError:
                print("Invalid command. Use: play <song_name>")
       
        elif command == "stop":
            subprocess.run(["powershell", "taskkill /im vlc.exe"], capture_output=True)
            print("Stopped playback.")
       
        elif command == "exit":
            print("Exiting...")
            await client.close()
            break
       
        else:
            print("Unknown command. Try: play <song_name>, stop, exit.")

@client.event
async def setup_hook():
    # Create the terminal input task during client setup
    asyncio.create_task(terminal_input())

BOT_TOKEN = "MTI5NjA5MzkyOTg5MTk1ODg0NQ.G4PC-C.MCgoFm5PPVAy3kIReU3L36lfkTQKdz1RCaPXB0"
client.run(BOT_TOKEN)
