import pandas as pd

DATA_PATH = "data/raw/Music info.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"

def clean_data(data):
    return (
        data
    .drop_duplicates( subset = "track_id")
    .drop( columns = ['genre', 'spotify_id'], errors='ignore')
    .fillna({'tags' : 'no_tags'})
    .assign(
        name = lambda x: x['name'].str.lower(),
        artists = lambda x: x['artist'].str.lower(),
        tags = lambda x: x['tags'].str.lower()
    )
    .reset_index(drop = True)
    )

def data_for_content_filtering(data):
    return ( 
        data.drop(columns = ['track_id', "name", "spotify_preview_url"],  errors='ignore')
    )

def main(data_path):
    data = pd.read_csv(DATA_PATH )
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(CLEANED_DATA_PATH ,index = False)
    

main(DATA_PATH)