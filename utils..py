from typing import Dict, List, Optional, Union, Tuple, BinaryIO  # Importing necessary types for type annotations
import os
import sys
import json
import tempfile
import copy
from tqdm.auto import tqdm
from functools import partial
from urllib.parse import urlparse
from pathlib import Path
import requests
from hashlib import sha256
from filelock import FileLock
import importlib_metadata
import torch
import torch.nn as nn
from torch import Tensor  # Tensor data type for PyTorch tensors
import fnmatch

__version__ = "4.0.0"  # Version of the current script
_torch_version = importlib_metadata.version("torch")  # Get the version of PyTorch installed

# Defining various cache directory paths for storing models and other files
hf_cache_home = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))
default_cache_path = os.path.join(hf_cache_home, "transformers")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

# Preset mirror URLs for downloading models
PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",  # Mirror for Tsinghua University
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",  # Mirror for Beijing Foreign Studies University
}

# Constants related to Hugging Face model handling
HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
WEIGHTS_NAME = "pytorch_model.bin"  # Default weight file name
CONFIG_NAME = "config.json"  # Default config file name

# Function to check if PyTorch is available
def is_torch_available() -> bool:
    """Returns True if PyTorch is available."""
    return True

# Function to check if TensorFlow is available
def is_tf_available() -> bool:
    """Returns False as TensorFlow is not used in this script."""
    return False

# Function to check if the provided URL is remote (HTTP/HTTPS)
def is_remote_url(url_or_filename: str) -> bool:
    """Returns True if the provided string is a remote URL."""
    parsed = urlparse(url_or_filename)  # Parse the URL
    return parsed.scheme in ("http", "https")  # Check if it's a HTTP or HTTPS URL

# Function for HTTP GET request with progress bar for downloading
def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
    """
    Download the content of a URL and save it to a temporary file.
    
    Args:
        url (str): URL to download.
        temp_file (BinaryIO): Temporary file object to write the content.
        proxies (dict, optional): Proxies to use for the request.
        resume_size (int, optional): The number of bytes already downloaded (for resuming).
        headers (dict, optional): Custom headers to use for the request.
    """
    headers = copy.deepcopy(headers)  # Make a copy of the headers dictionary
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)  # Add Range header if resuming download
    
    # Make the HTTP request
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()  # Raise exception for bad responses
    content_length = r.headers.get("Content-Length")  # Get content length from response headers
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B", unit_scale=True, total=total, initial=resume_size, desc="Downloading", disable=False
    )  # Create a progress bar for download
    for chunk in r.iter_content(chunk_size=1024):  # Iterate over the content in chunks
        if chunk:  # Filter out keep-alive new chunks
            progress.update(len(chunk))  # Update the progress bar
            temp_file.write(chunk)  # Write the chunk to the temporary file
    progress.close()

# Function to generate a unique filename for a given URL
def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Generates a unique filename based on the URL and optionally the ETag.
    
    Args:
        url (str): The URL for which the filename is generated.
        etag (str, optional): ETag value to make the filename unique.
    
    Returns:
        str: The generated filename.
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()  # Hash the URL using SHA256

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()  # If an ETag is provided, hash it and append it to the filename

    if url.endswith(".h5"):
        filename += ".h5"  # Append .h5 if the URL ends with it

    return filename

# Function to build a Hugging Face bucket URL for a specific model and file
def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Generates the URL for a model file in the Hugging Face bucket.
    
    Args:
        model_id (str): The model ID.
        filename (str): The file name to download.
        subfolder (str, optional): Subfolder where the file is located.
        revision (str, optional): Model revision (default is "main").
        mirror (str, optional): The mirror to use for downloading.
    
    Returns:
        str: The generated URL for the model file.
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"  # Include subfolder if present

    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)  # Get the mirror URL
        legacy_format = "/" not in model_id  # Determine if the model ID is in legacy format
        if legacy_format:
            return f"{endpoint}/{model_id}-{filename}"
        else:
            return f"{endpoint}/{model_id}/{filename}"

    if revision is None:
        revision = "main"  # Default to main revision
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)

# Function to get a user agent string for HTTP requests
def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Generates a user-agent string for HTTP requests based on available libraries.
    
    Args:
        user_agent (Union[Dict, str, None], optional): Custom user agent string or dictionary.
    
    Returns:
        str: The generated user agent string.
    """
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())  # Append key-value pairs
    elif isinstance(user_agent, str):
        ua += "; " + user_agent  # Append the string if provided
    return ua

# Function to retrieve a model file from cache or download it
def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Downloads and caches a model file if not already present in cache.
    
    Args:
        url (str): URL to download the file from.
        cache_dir (str, optional): Directory to store the cached file.
        force_download (bool, optional): Whether to force re-download.
        proxies (dict, optional): Proxies for the download.
        etag_timeout (int, optional): Timeout for ETag.
        resume_download (bool, optional): Whether to resume a previous download.
        user_agent (Union[Dict, str, None], optional): Custom user agent for the request.
        use_auth_token (Union[bool, str, None], optional): Authorization token.
        local_files_only (bool, optional): Whether to restrict to local files only.
    
    Returns:
        Optional[str]: The path to the cached model file.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE  # Default cache directory
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)  # Create the cache directory if it doesn't exist

    headers = {"user-agent": http_user_agent(user_agent)}  # Set up headers for the HTTP request
    if isinstance(use_auth_token, str):
        headers["authorization"] = "Bearer {}".format(use_auth_token)  # Add authorization token if provided
    elif use_auth_token:
        token = HfFolder.get_token()  # Get token from HuggingFace folder
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = "Bearer {}".format(token)  # Use the token for authorization

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")  # Get the ETag from the response headers
            if etag is None:
                raise OSError("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.")
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]  # Follow redirects if needed
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Handle connection errors or timeouts

    filename = url_to_filename(url, etag)  # Generate the filename from the URL

    # Get cache path to save the file
    cache_path = os.path.join(cache_dir, filename)

    if etag is None:
        if os.path.exists(cache_path):
            return cache_path  # Return cached file if exists
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])  # Return the matching file if found
            else:
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only' to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # If we have the etag, proceed with download
    if os.path.exists(cache_path) and not force_download:
        return cache_path  # Return cached file if exists and not forcing a download

    # Prevent parallel downloads of the same file with a lock
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path  # Return cached file if it already exists

        # Download file if not already downloaded
        if resume_download:
            incomplete_path = cache_path + ".incomplete"
            temp_file_manager = _resumable_file_manager  # Handle resumable downloads
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size  # Get the size of the incomplete download
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to a temporary file, then move to the cache directory
        with temp_file_manager() as temp_file:
            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

        os.replace(temp_file.name, cache_path)  # Replace the incomplete file with the final downloaded file

        # Save metadata (URL and ETag) for caching purposes
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)  # Write metadata to a JSON file

    return cache_path  # Return the path to the cached file

# Function to handle compressed files (like zip or tar)
def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Retrieves the cached path for a given URL or filename, optionally extracting compressed files.
    
    Args:
        url_or_filename (str): URL or local filename to retrieve.
        cache_dir (str, optional): Directory to store the cache.
        force_download (bool, optional): Whether to force a download.
        proxies (dict, optional): Proxies for the request.
        resume_download (bool, optional): Whether to resume a download.
        user_agent (Union[Dict, str, None], optional): Custom user agent for HTTP requests.
        extract_compressed_file (bool, optional): Whether to extract compressed files (e.g., .zip, .tar).
        force_extract (bool, optional): Force extraction even if extracted files exist.
        use_auth_token (Union[bool, str, None], optional): Authorization token.
        local_files_only (bool, optional): Whether to restrict to local files only.
    
    Returns:
        Optional[str]: The path to the cached file or extracted directory.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE  # Use default cache path if not provided
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)  # Convert Path to string if it's a Path object
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        

        # If URL, download and get the cached path
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # If local file exists, use it
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # Raise error if the file doesn't exist locally
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Raise error for unknown formats
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path  # If not a valid compressed file, return path directly

        # Handle extraction of compressed files (zip/tar)
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted  # Return extracted directory if it exists

        # Prevent parallel extractions with a lock
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)  # Remove old extracted files
            os.makedirs(output_path_extracted)  # Create new directory for extraction
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)  # Extract zip file
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)  # Extract tar file
                tar_file.close()
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted  # Return the path of extracted files

    return output_path  # Return the path to the original file

# Function to get the data type of the model parameters (tensors)
def get_parameter_dtype(parameter: nn.Module) -> torch.dtype:
    try:
        return next(parameter.parameters()).dtype  # Get dtype of the first parameter in the module
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]  # Find tensor attributes
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)  # Get tensor attributes
        first_tuple = next(gen)  # Get the first tuple of the parameter
        return first_tuple[1].dtype  # Return the dtype of the tensor

# Function to get an extended attention mask (used in attention layers of Transformers)
def get_extended_attention_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    # attention_mask has shape [batch_size, seq_length]
    assert attention_mask.dim() == 2  # Ensure it's a 2D tensor
    # Extend attention mask for multi-head attention
    extended_attention_mask = attention_mask[:, None, None, :]  # Shape becomes [batch_size, 1, 1, seq_length]
   
  extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
  extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
  return extended_attention_mask