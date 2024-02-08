**Similar Influencers Project Rundown**

Goal: To help customers find influencers they want to work with, dependent on their likes/dislikes of other influencers

Method: We use image and sentence embeddings to generate influencer level vectors and apply a Pinecone similarity search.

Use case: Customers often provide examples of influencers they like, which are used as a benchmark to find similar influencers. This automates this process, improving efficiency and precision in influencer recommendations.

**Code Utilization**

Run the file _Similar Influencers Nice Dash.py _ to visualize the Dash app. Choose yes if you like the influencer, no if you don't, easy as that.

_Similar Influencers Vector Generation.py_ details how the vectors were generated.

All the JSON files for the ~4,000 influencer vectors created are in the notebook/AvroDataJson folder
