import pickle
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer
import requests
import time
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML
from tqdm import tqdm

import os
import sys
import glob
import json


# from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
# import tensorflow as tf
# import tensorflow_hub as hub
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
from sklearn.decomposition import PCA
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

from pinecone import Pinecone

pc = Pinecone(api_key="55616add-2e3f-4b54-8907-ac36421b6b58")
pineconeindex = pc.Index("influencer-vectors")



# import emoji

#this portion can be replaced by whatever list of guids you wish to make embeddings for as long as you have them in the notebook or can access them programattically

# guid_values = qual_inf['guid'].tolist()
with open('data_guids.txt', 'r') as file:
    # Read all the lines of the file into a list
    guid_values = [line.strip() for line in file.readlines()]
# print(guid_values)



embeds = []
index = 0

for val_guid in range(len(guid_values)):
# for guid in tqdm(guid_values, desc = "Processing Influencers", unit = "Influencers"):
    guid = guid_values[val_guid+29]
    index += 1
    if index == 5: break

    # guid= sza
#     print(guid)
    TIME_PERIOD = 12 # do not change this 
    file_name = str(TIME_PERIOD)+'_'+guid+'.json'

    #this can be replaced by the download avro function from linqia jupyter if accessing data from aws directly
    f = open("notebook/AvroDataJson/" + guid + '.json')
    avro_data = json.load(f)



    social_posts = avro_data['social_posts']
    count = 0
    count2 = 0
    influencer_embedding = np.zeros(1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Use tqdm to add a progress bar
    for post in tqdm(social_posts, desc="Processing social posts", unit="post"):
        
        
        caption = post['post_text']

        
        media = post['media']
        imagefeatures = np.zeros(512)
        if len(media)==0: 
          print("no image")
          continue
        image_count=0
        for i in range(len(media)):
        
            media_count = media[i]

            image_url = media_count['url']

            
            try:
              response = requests.get(image_url)
              image = Image.open(BytesIO(response.content))
            except:
              continue
 


            image1 = preprocess(image).unsqueeze(0).to(device)
            
            imagefeatures += model.encode_image(image1).detach().numpy().flatten()
            image_count += 1

            
        if image_count > 0: imagefeatures = imagefeatures/image_count

        count2 += 1


        try:
            context_length = 77

            blocks = [caption[i:i+context_length] for i in range(0, len(caption), context_length)]
            
            embeddings = [clip.tokenize(block, context_length=context_length).to(device) for block in blocks]
            

        except:
            continue

        textfeaturesall = [model.encode_text(embeds) for embeds in embeddings]
        textfeatures = torch.stack(textfeaturesall).mean(dim=0).to(device)
        count += 1


        
        stacked = np.hstack([imagefeatures,np.mean(textfeatures.detach().numpy(),axis=0).flatten()])

        influencer_embedding = influencer_embedding + stacked
        
    if count2 > 0:
        influencer_embedding[:len(influencer_embedding)//2] /= count2
    else:
        influencer_embedding[:len(influencer_embedding)//2] /= count
        
    influencer_embedding[len(influencer_embedding)//2:] /= count
    embeds.append(influencer_embedding)


    pineconeindex.upsert([{"id":guid,"values":influencer_embedding.tolist()}])
    




first_guids = guid_values[:index-1]

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeds, embeds)

# Create a DataFrame with the cosine similarity matrix and set the first 5 guids as indices
df_cosine_similarity = pd.DataFrame(cosine_sim_matrix, index=first_guids, columns=first_guids)

# Print or save the DataFrame as needed
df_cosine_similarity.to_csv('firstCLIPSimilarity.csv')


np.save("embeds_array.npy", embeds)

loaded_embeds = np.load("embeds_array.npy")

print(np.shape(loaded_embeds))


