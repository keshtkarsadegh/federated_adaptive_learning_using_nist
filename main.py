from src.federated_adaptive_learning_nist.nist_downloader_extractor import download_and_extract_with_progress

if __name__ == "__main__":
    download_and_extract_with_progress("https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",extract_to="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data")
    download_and_extract_with_progress("https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",extract_to="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data")
