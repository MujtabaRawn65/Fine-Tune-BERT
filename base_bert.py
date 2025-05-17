import re
from torch import device, dtype
from config import BertConfig, PretrainedConfig
from utils import *

class BertPreTrainedModel(nn.Module):
    ## Set the default configuration class for this model
    # This will define the configuration used when instantiating the model
    config_class = BertConfig
    
    ## This defines the root attribute of your model (e.g., model.bert for BERT-based models)
    # The base model prefix refers to the main architecture of the model (e.g., 'bert' for BERT)
    base_model_prefix = "bert"
    
    ## Keys to ignore when missing in state_dict loading
    # These keys will be ignored during the loading of the model's state_dict if they are missing
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    ## Keys to ignore when unexpected keys are found during state_dict loading
    # These keys will be ignored if found unexpectedly in the loaded state_dict
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        """
        Initializes the model with the given configuration.
        
        Args:
            config (PretrainedConfig): The configuration for the model (usually BERTConfig).
            *inputs: Any additional inputs to initialize the model (usually tokenizers, layers, etc).
            **kwargs: Additional keyword arguments for model initialization.
        """
        super().__init__()
        
        ## Save the configuration object for later use in the model
        self.config = config
        
        ## Store the model path or name in case it is needed for downloading weights or inference
        self.name_or_path = config.name_or_path

    def init_weights(self):
        """
        Initialize the weights of the model using the _init_weights method.
        """
        # Apply the weight initialization recursively across all modules (layers).
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Handles the initialization of weights for different types of layers in the model.
        
        Args:
            module (nn.Module): The layer/module being initialized.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            ## For Linear and Embedding layers, initialize weights using normal distribution
            # Mean = 0.0, std = initializer_range from the configuration
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        elif isinstance(module, nn.LayerNorm):
            ## For LayerNorm layers, initialize biases to zero and weights to 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # If the module is a Linear layer and it has a bias term, initialize it to zero
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> dtype:
        """
        Get the dtype of the model parameters (for example, float32).
        
        Returns:
            dtype: The data type of the model's parameters.
        """
        return get_parameter_dtype(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        """
        Load the pretrained model from a path or model hub.
        
        Args:
            pretrained_model_name_or_path (str or PathLike): Path to the model or its name on the Hugging Face model hub.
            *model_args: Additional arguments passed during model instantiation.
            **kwargs: Additional keyword arguments (e.g., for caching, proxies, etc).
        
        Returns:
            model: A model instance loaded with pretrained weights.
            loading_info (optional): Information about missing/unexpected keys in the loaded state_dict.
        """
        # Extract common kwargs for loading the model and configuration
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)

        # If the configuration is not provided, load it from the model's path or name
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        ## Resolve path or URL to model weights
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                # If a directory is provided, look for weight file inside the directory
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                # If it's a direct file path or a URL, use that as the archive file
                archive_file = pretrained_model_name_or_path
            else:
                # Otherwise, download from Hugging Face model hub
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME,
                    revision=revision,
                    mirror=mirror,
                )
            
            try:
                ## Download or load model weights from the resolved archive file
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError as err:
                ## If loading fails, raise an error with a helpful message
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)
        else:
            resolved_archive_file = None

        # Set the model's path in the configuration object
        config.name_or_path = pretrained_model_name_or_path

        ## Instantiate the model using the configuration and any additional arguments
        model = cls(config, *model_args, **model_kwargs)

        # If no state_dict is provided, try loading from the cached weights
        if state_dict is None:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                    f"at '{resolved_archive_file}'"
                )

        # Initialize missing/unexpected key tracking
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Map old state_dict keys to new keys, in case of architecture changes
        old_keys = []
        new_keys = []
        m = {
            'embeddings.word_embeddings': 'word_embedding',
            'embeddings.position_embeddings': 'pos_embedding',
            'embeddings.token_type_embeddings': 'tk_type_embedding',
            'embeddings.LayerNorm': 'embed_layer_norm',
            'embeddings.dropout': 'embed_dropout',
            'encoder.layer': 'bert_layers',
            'pooler.dense': 'pooler_dense',
            'pooler.activation': 'pooler_af',
            'attention.self': "self_attention",
            'attention.output.dense': 'attention_dense',
            'attention.output.LayerNorm': 'attention_layer_norm',
            'attention.output.dropout': 'attention_dropout',
            'intermediate.dense': 'interm_dense',
            'intermediate.intermediate_act_fn': 'interm_af',
            'output.dense': 'out_dense',
            'output.LayerNorm': 'out_layer_norm',
            'output.dropout': 'out_dropout'
        }

        # Apply the renaming rules to convert old state_dict keys to the new format
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            for x, y in m.items():
                if new_key is not None:
                    _key = new_key
                else:
                    _key = key
                if x in key:
                    new_key = _key.replace(x, y)
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        # Replace old keys with the newly renamed keys
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Copy state_dict to maintain metadata information
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # Check for unknown parameters and raise an error if needed
        your_bert_params = [f"bert.{x[0]}" for x in model.named_parameters()]
        for k in state_dict:
            if k not in your_bert_params and not k.startswith("cls."):
                possible_rename = [x for x in k.split(".")[1:-1] if x in m.values()]
                raise ValueError(f"{k} cannot be loaded into your model, some parameters may have been renamed")

        # Recursively load the state_dict into the model
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Handle prefix logic for base models (e.g., only `bert` exists in the dict)
        start_prefix = ""
        model_to_load = model
        has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)

        # Load the state_dict recursively into the model
        load(model_to_load, prefix=start_prefix)

        # Add missing keys related to the base model if the head model has different parameters
        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]
            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        # Remove allowed missing/unexpected keys based on regex patterns
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Raise an error if there are any loading errors
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        # Set the model to evaluation mode by default
        model.eval()

        # If requested, return loading info along with the model
        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        # If using TPU, move model to the appropriate device (XLA)
        if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model
