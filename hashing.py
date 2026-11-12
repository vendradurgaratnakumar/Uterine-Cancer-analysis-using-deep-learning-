"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                            HASHING
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""


import numpy as np
from scipy.stats import shapiro
import hashlib


# STEP 1: WEIGHT SELECTION 
def weight_selection(feature_vector, keep_ratio=4):

    abs_feat = np.abs(feature_vector)
    threshold = np.quantile(abs_feat, 1 - (1 / keep_ratio))
    selected = np.where(abs_feat >= threshold, feature_vector, 0.0)

    return selected



# STEP 2: NORMALIZATION
def normalize_features(feature_vector):

    mean = np.mean(feature_vector)
    std = np.std(feature_vector)

    normalized = (feature_vector - mean) / (std + 1e-8)

    return normalized



# STEP 3: NTS FEATURE GENERATION
def compute_nts(feature_vector, num_segments=64):

    segments = np.array_split(feature_vector, num_segments)

    nts_values = []

    for seg in segments:
        seg = np.asarray(seg).flatten()

        if len(seg) < 3:
            nts_values.append(0.0)
            continue

        try:
            statistic, _ = shapiro(seg)

            nts_values.append(
                float(statistic)
            )

        except:
            nts_values.append(0.0)

    return np.asarray(nts_values, dtype=np.float32)



# STEP 4: QUANTIZATION 
def quantize_nts(nts_values):

    hash_bits = np.zeros(len(nts_values) - 1, dtype=np.uint8)

    for i in range(len(nts_values) - 1):
        if nts_values[i] > nts_values[i + 1]:
            hash_bits[i] = 1
        else:
            hash_bits[i] = 0

    return hash_bits



# STEP 5: HASHING
def encrypt_hash(hash_bits, secret_key=2025):

    rng = np.random.RandomState(secret_key)
    key_bits = rng.randint(0, 2, size=len(hash_bits), dtype=np.uint8)
    encrypted_hash = np.bitwise_xor(hash_bits, key_bits)

    return encrypted_hash



# GENERATE SINGLE HASH
def generate_single_npsh_hash(feature_vector, keep_ratio=4, num_segments=64, secret_key=2025):

    selected_features = weight_selection(feature_vector, keep_ratio)
    
    normalized_features = normalize_features(selected_features)
    nts_values = compute_nts(normalized_features, num_segments)

    binary_hash = quantize_nts(nts_values)
    feature_vector = np.round(feature_vector, decimals=4)

    feature_bytes = (np.asarray(feature_vector).astype(np.float32).tobytes())
    unique_string = (feature_bytes + str(secret_key).encode())

    sha_hash = hashlib.sha256(unique_string).digest()
    sha_bits = np.unpackbits(np.frombuffer(sha_hash,dtype=np.uint8))

    final_hash = np.concatenate([binary_hash, sha_bits[:64]])

    return final_hash.astype(np.uint8)



# GENERATE HASHES FOR ENTIRE DATASET
def generate_npsh_hashes(deep_features, keep_ratio=4, num_segments=64, secret_key=2025):

    all_hashes = []

    total_samples = len(deep_features)
    print("Generating NPSH Hashes...\n")

    for idx, feature_vector in enumerate(deep_features):
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx+1}/{total_samples}")

        secure_hash = generate_single_npsh_hash(
            feature_vector,
            keep_ratio=keep_ratio,
            num_segments=num_segments,
            secret_key=secret_key
        )
        all_hashes.append(secure_hash)

    all_hashes = np.asarray(all_hashes, dtype=np.uint8)

    return all_hashes











