import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import pickle
import pandas as pd
import numpy as np
import requests
import time
import traceback
from IPython.display import HTML
from tqdm import tqdm

import os
import sys
import glob
import json

from numpy.linalg import norm
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from pinecone import Pinecone
import numpy as np

pc = Pinecone(api_key="55616add-2e3f-4b54-8907-ac36421b6b58")
index = pc.Index("influencer-vectors")
print("Pinecone Index created successfully!")



app = dash.Dash(__name__)


guid = "00004a5a-da8f-4924-b4b6-e0da6097caee"
# guid = "00091885-70d1-4833-a482-ed457de08bb8"
# guid = "0045b824-76bf-4160-a770-45d8855db1be"

file_path = "AU_Interns_2024/Similar Influencers/notebook/AvroDataJson/"

current_profile = guid
f = open(file_path + guid + '.json')
avro_data = json.load(f)
handle = avro_data["influencer_handle"]
pic_url = avro_data["profile_picture_url"]
social_posts = avro_data['social_posts']

post_index = -1

first = social_posts[post_index]
caption = first['post_text']
media = first['media']
media_first = media[0]
image = media_first['url']

similarity = 100

app.layout = html.Div(
    style={'text-align': 'center', 'backgroundColor': '#2B2B2B', 'color': 'white', 'font-family': 'Arial, sans-serif'},  # Center-align the content
    children=[
        html.Div(
            children=[
                        html.A(
                            dcc.Link(
                                "@" + handle,
                                href=f"https://www.instagram.com/{handle}",
                                target="_blank",
                                style={
                                    'margin-bottom': '10px',
                                    'font-size': '18px',
                                    'color': 'white !important',
                                    'text-decoration': 'none'  # Removing underline
                                },
                                id="handle-output"
                            ),
                            style={'color': 'white'}  # Overriding default link color
                        ),                
                        html.Img(src=pic_url, alt="Profile Photo", id="profile-photo", style={'display': 'block', 'margin': 'auto', 'border-radius': '50%', 'width': '120px', 'height': '120px'}),
            ]
        ),
        html.Div(style={'margin-top': '20px'}),  # Add space between images
        html.Div(
            children=[
                html.H1("👯 TwinFluencers 👯", style={'margin-top': '10px', 'margin-bottom': '10px','font-size': '20px'}),
                html.Div([
                    html.Button('←', id='left-post-button', n_clicks=0, style={'font-size': '30px', 'border': 'none', 'background-color': 'black', 'color': 'white', 'position': 'absolute', 'left': '0', 'top': '50%', 'transform': 'translateY(-50%)'}),
                    html.Img(src=pic_url, alt="Post", id="post", style={'width': '100%', 'height': '100%', 'object-fit': 'cover', 'border-radius': '10px'}),
                    html.Button('→', id='right-post-button', n_clicks=0, style={'font-size': '30px', 'border': 'none', 'background-color': 'black', 'color': 'white', 'position': 'absolute', 'right': '0', 'top': '50%', 'transform': 'translateY(-50%)'}),
                ], style={'position': 'relative', 'width': '320px', 'height': '320px', 'margin': 'auto', 'overflow': 'hidden'}),
                html.P("Caption: " + caption, style={'margin': 'auto','margin-top': '10px', 'font-size': '16px', 'height': '80px', 'width' : '320px', 'overflow': 'auto'}, id="post-caption"),
            ],
            style={'position': 'relative'}  # Add relative positioning to the container div
        ),
        html.Div(style={'margin-top': '20px'}),  # Add space between images
        html.Div([
            html.Button('←', id='left-arrow-button', n_clicks=0, style={'font-size': '30px', 'border': 'none', 'background-color': 'transparent', 'color': 'white'}),
            html.Button('→', id='right-arrow-button', n_clicks=0, style={'font-size': '30px', 'border': 'none', 'background-color': 'transparent', 'color': 'white'}),
            html.P(id='arrow-output', style={'margin-top': '10px', 'font-size': '14px'})  # Display the arrow click feedback
        ]),
        html.Div(
            id="similarity-bubble",
            children=[
                html.Div(id="similarity-text", style={'font-size': '16px', 'color': 'white', 'padding': '10px', 'border-radius': '20px', 'background-color': '#1E90FF', 'position': 'fixed', 'right': '20px', 'top': '20px'}),
            ]
        ),
        html.Div(
            style={'position': 'absolute', 'left': '50%', 'transform': 'translateX(-50%)', 'bottom': '20px'},  # Center the buttons
            children=[
                html.Button("Yes", id="yes-button", n_clicks=0, style={'border-radius': '50%', 'background-color': '#3897f0', 'border': '2px solid #3897f0', 'color': 'white', 'font-size': '18px', 'padding': '10px', 'margin-right': '20px'}),
                html.Button("No", id="no-button", n_clicks=0, style={'border-radius': '50%', 'background-color': '#ed4956', 'border': '2px solid #ed4956', 'color': 'white', 'font-size': '18px', 'padding': '10px'}),
            ]
        ),
        html.Div(id="selected-handles-output")
    ]
)

selected_handles = []

yes_guids = []
no_guids = []


def update_profile_index(current_guid, yes_guids, no_guids):


    fetched_yes = index.fetch(ids = yes_guids)
    # fetched_no = index.fetch(ids = no_guids)
    
    yes_vector_list = []

    yes_vectors = fetched_yes['vectors']
    for id in yes_guids:
        vector = yes_vectors[id]['values']
        yes_vector_list.append(vector)

    yes_vector_np = np.array(yes_vector_list)
    average_vector_yes = np.mean(yes_vector_np, axis=0)


    average_vector_no = np.zeros(1024)

    if len(no_guids) > 0:
        fetched_no = index.fetch(ids = no_guids)  
        
        no_vector_list = []

        no_vectors = fetched_no['vectors']
        for id in no_guids:
            vector = no_vectors[id]['values']
            no_vector_list.append(vector)

        no_vector_np = np.array(no_vector_list)
        average_vector_no = np.mean(no_vector_np, axis=0)/2


    average_vector = average_vector_yes - average_vector_no


    num_queries = len(yes_guids) + len(no_guids) + 1

    results = index.query(vector=average_vector.tolist(), top_k=num_queries) 

    people = results['matches']
    for person in people:
        if person['id'] not in yes_guids and person['id'] not in no_guids:
            current_guid = person['id']
            print(current_guid)
            similarity = np.floor(person['score']*100)
            print(similarity)
            break

    

    return (current_guid,similarity)



@app.callback(
    [Output("profile-photo", "src"),
     Output("post", "src"),
     Output("profile-photo", "alt"),
     Output("handle-output", "children"),  # New output for handle
     Output("post-caption", "children"),  # New output for caption
     Output("selected-handles-output", "children"),
     Output('similarity-text',"children")
     ],
    [Input("yes-button", "n_clicks"),
     Input("no-button", "n_clicks"),
     Input("right-arrow-button", "n_clicks"),
     Input("left-arrow-button", "n_clicks"),
     Input("right-post-button", "n_clicks"),
     Input("left-post-button", "n_clicks")],  # Separate input for left arrow button
    [State("selected-handles-output", "children")]
)
def update_profile(n_clicks_yes, n_clicks_no, n_clicks_next, n_clicks_prev, n_clicks_right, n_clicks_left,  output_text):
    global current_profile
    global selected_handles
    global handle
    global guid
    global caption
    global post_index
    global image_index
    global image
    global yes_guids
    global no_guids
    global similarity

    # Initialize selected_handles, handle, and caption if not already initialized
    if 'selected_handles' not in globals():
        selected_handles = []
    if 'handle' not in globals():
        handle = ""
    if 'caption' not in globals():
        caption = ""
    if 'post_index' not in globals():
        post_index = -1
    if 'image_index' not in globals():
        image_index = 0
    if 'similarity' not in globals():
        similarity = 100


    # Determine which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered_id
    if button_id == "yes-button":
        # Add the current profile handle to the list
        #selected_handles = output_text.split(":")[1].strip().split(", ")
        selected_handles.append(handle)
        yes_guids.append(guid)

        (current_profile,similarity) = update_profile_index(current_profile, yes_guids, no_guids)

        post_index = -1

    elif button_id == "no-button":
        no_guids.append(guid)
        (current_profile,similarity) = update_profile_index(current_profile, yes_guids, no_guids)
        post_index = -1

    # guid = guid_values[current_profile]
    guid = current_profile

    f = open(file_path + guid + '.json')
    avro_data = json.load(f)
    handle = avro_data["influencer_handle"]
    pic_url = avro_data["profile_picture_url"]
    social_posts = avro_data["social_posts"]
    if button_id == "right-arrow-button":
        # Increment post index and wrap around
        post_index = (post_index + 1) % len(social_posts)
    
    if button_id == "left-arrow-button":
        # Decrement post index and wrap around
        post_index = (post_index - 1) % len(social_posts)
        
    # Update the profile photo source
    profile_photo_src = pic_url
    # Update the post details
    current_post = social_posts[post_index]
    caption = current_post['post_text']

    media_present = 'media' in current_post and current_post['media']

    if media_present:
        image = current_post['media'][image_index]['url']

    # if media_present and button_id == "right-post-button":
    #     if current_post['media'][image_index + 1]:
    #         image_index = (image_index + 1)
    # if media_present and button_id == "left-post-button":
    #     if current_post['media'][image_index - 1]:
    #         image_index = (image_index - 1)
    
    similarity_text = f"Similarity: {similarity}%"

    handle_link = html.A("@"+handle, href=f"https://www.instagram.com/{handle}", target="_blank", id="handle-output", style={'margin-bottom': '10px'})

    # Update the output text
    selected_handles_text = ' '.join(f'@{handle}' for handle in selected_handles)
    handle_counter_text = f" ({len(selected_handles)})" if selected_handles else ""
    output_text = [f"Selected handles: {selected_handles_text}", html.Br(), handle_counter_text]

    return profile_photo_src, image, handle, handle_link, f"Caption: {caption}", output_text, similarity_text

# Running the app
if __name__ == "__main__":
    app.run_server(debug=True)