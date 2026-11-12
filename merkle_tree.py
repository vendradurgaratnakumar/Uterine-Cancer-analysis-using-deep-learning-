"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                          Merkle Tree
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""


import hashlib
import numpy as np
import math


# Hash Function
def H(data):
    hsh = hashlib.sha256(data).digest()
    return hsh


# Leaf Digest
def create_leaf(hash_value):
    tag_L = b'\x00'
    L = H(tag_L + hash_value.astype(np.uint8).tobytes())
    return L


# Internal Node
def create_parent(left, right):
    tag_I = b'\x01'
    parent = H(tag_I + left + right)
    return parent


# Padding
def pad_tree(leaves):
    
    n = len(leaves)
    target = 2 ** math.ceil(math.log2(n))

    while len(leaves) < target:
        lf = H(b'\x00' + b'0')
        leaves.append(lf)

    return leaves



# Construct EAMT Tree
def build_tree(leaves):

    levels = []
    current = leaves.copy()
    
    # store leaves
    levels.append(current)

    while len(current) > 1:
        parents=[]
        
        for i in range(0, len(current), 2):
            parent=create_parent(current[i], current[i+1])
            parents.append(parent)

        current = parents
        levels.append(current)

    return levels[::-1]



# Ancestor Checkpoints
def generate_ancestors(tree, stride):

    ancestors={}
    nodes = tree[stride]

    for k,node in enumerate(nodes):

        ancestors[k] = {
            "ancestor_id":k,
            "level":stride,
            "hash":node.hex()
        }

    return ancestors



# Upward Proof
def ancestor_proof(tree, index, stride):

    proof=[]
    position=index

    for level in range(stride, len(tree)-1):

        sibling = (
            position+1 if position%2==0
            else
            position-1
        )

        proof.append(tree[level+1][sibling].hex())
        position//=2

    return proof



# Ancestor Certificate
def create_certificates(tree, stride):

    certificates={}
    ancestors = generate_ancestors(tree, stride)

    for k,a in ancestors.items():

        certificates[k] = {
            "ancestor_hash":a["hash"],
            "stride":stride,
            "index":k,
            "upProof":ancestor_proof(tree, k, stride),
            "sequence":k
        }

    return certificates



# Epoch Aggregation
def aggregate_root(root):
    
    rt = H(root).hex()
    
    return rt




def EAMT(hashes, stride=4):

    print("Generating Leaf Digests...")
    leaves=[]

    for h in hashes:
        leaves.append(create_leaf(h))

    leaves = pad_tree(leaves)

    print("Building EAMT...")
    tree=build_tree(leaves)

    # Root
    root_hash = tree[0][0].hex()

    print("Generating Ancestor Certificates...")
    certificates=create_certificates(tree, stride)

    # Blockchain commitment
    blockchain_root = aggregate_root(bytes.fromhex(root_hash))

    EAMT_record={
        "Total_Records":len(hashes),
        "Tree_Depth":len(tree)-1,
        "Stride":stride,
        "Root_Hash":root_hash,
        "Blockchain_Commit":blockchain_root,
        "Ancestor_Certificates":certificates
    }

    return EAMT_record, tree, root_hash









