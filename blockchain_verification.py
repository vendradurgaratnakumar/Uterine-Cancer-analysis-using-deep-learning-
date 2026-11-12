import json
import numpy as np



def verify(hashes):

    with open("Features/blockchain_record.json", "r") as f:
        blockchain = json.load(f)

    blockchain_hashes = np.array(blockchain["NPSH_Hashes"])
    ipfs_hashes = np.array(blockchain["IPFS_Hashes"])
    keys = blockchain["Keys"] 

    retrieved_ipfs_list = []
    retrieved_keys = []         

    for i, h in enumerate(hashes):

        match_index = np.where(np.all(blockchain_hashes == h, axis=1))[0]

        if len(match_index) > 0:

            bc_index = match_index[0]

            retrieved_ipfs = ipfs_hashes[bc_index]
            retrieved_key = keys[bc_index]   

            retrieved_ipfs_list.append(retrieved_ipfs)
            retrieved_keys.append(retrieved_key)

            print(f"Hash {i+1} - Access Granted")


        else:
            print(f"Hash {i+1} - Access Denied")

            raise SystemExit(
                f"Process stopped: Unauthorized hash detected at Hash {i+1}"
            )


    return np.array(retrieved_ipfs_list), retrieved_keys





import json
import requests


" Retrieve Image From IPFS "

def get_from_ipfs(ipfs_hash):
    
    url = f"http://127.0.0.1:5001/api/v0/cat?arg={ipfs_hash}"
    response = requests.post(url)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")
 
    
def download_encrypted_data_from_ipfs(ipfs_hash_list):
    encrypted_data = []
    
    for index, ipfs_hash in enumerate(ipfs_hash_list):
        try:
            if ipfs_hash is not None:
                data = get_from_ipfs(ipfs_hash)
                encrypted_data.append(data)
                # print(f"Downloaded {index + 1}/{len(ipfs_hash_list)}")
            else:
                print(f"Hash missing for image {index}")
                encrypted_data.append(None)
        except Exception as e:
            print(f"Error downloading {ipfs_hash}: {e}")
            encrypted_data.append(None)

    return encrypted_data




















