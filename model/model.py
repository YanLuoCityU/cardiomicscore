import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Type, Optional, Dict, Any

# Assume _resolve_pytorch_module is defined as in the previous responses:
def _resolve_pytorch_module(module_spec: Union[str, Type[nn.Module], None],
                            default_module_prefix: str = "torch.nn") -> Optional[Type[nn.Module]]:
    if not module_spec: return None
    if isinstance(module_spec, type) and issubclass(module_spec, nn.Module): return module_spec
    if not isinstance(module_spec, str):
        raise TypeError(f"module_spec must be a string or nn.Module class, got {type(module_spec)}")
    parts = module_spec.split('.')
    if len(parts) == 1: class_name = parts[0]; module = torch.nn if default_module_prefix == "torch.nn" else None
    elif len(parts) == 2 and parts[0] == "nn": class_name = parts[1]; module = torch.nn
    else: raise ValueError(f"Unsupported module string format: {module_spec}.")
    if module is None: raise ValueError(f"Module for {module_spec} could not be resolved.")
    try: return getattr(module, class_name)
    except AttributeError: raise ValueError(f"Module class '{class_name}' not found in '{module.__name__}'.")

# The refactored MLP class (from previous response, ensure it's defined)
class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int, # This is the final output dimension of this MLP (after its head)
                 hidden_dims: Optional[List[int]] = None, # List of hidden layer sizes, e.g., [256, 128]
                 activation_cls_str: str = "nn.ReLU", # e.g., "nn.ReLU"
                 norm_cls_str: Optional[str] = "nn.BatchNorm1d", # e.g., "nn.BatchNorm1d", "nn.LayerNorm", or None
                 apply_norm_to_layers: Union[str, List[int]] = "all", # "all", "none", or list of 0-indexed layer indices
                 input_norm: bool = False, # Apply LayerNorm to the input
                 dropout_p: float = 0.5,
                 final_activation_cls_str: Optional[str] = None, # Activation for the very final output
                 final_norm: bool = False, # Apply norm to the output of the head
                 final_dropout: bool = False, # Apply dropout to the output of the head
                 custom_init: bool = False): # Flag for custom weight initialization
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim # Final output dim of this MLP instance
        self.dropout_p = dropout_p

        activation_cls = _resolve_pytorch_module(activation_cls_str)
        norm_cls = _resolve_pytorch_module(norm_cls_str)
        final_activation_cls = _resolve_pytorch_module(final_activation_cls_str)

        hidden_dims_list = hidden_dims if hidden_dims is not None else []

        self.input_norm_layer = nn.LayerNorm(input_dim) if input_norm else None

        # Build the main MLP layers (feature extractor part)
        mlp_layers_list = []
        current_feature_dim = input_dim
        
        num_hidden_actual = len(hidden_dims_list)
        norm_indices = []
        if norm_cls:
            if isinstance(apply_norm_to_layers, str):
                if apply_norm_to_layers == "all" and num_hidden_actual > 0:
                    norm_indices = list(range(num_hidden_actual))
                elif apply_norm_to_layers == "none":
                    norm_indices = []
            elif isinstance(apply_norm_to_layers, list):
                norm_indices = [idx for idx in apply_norm_to_layers if 0 <= idx < num_hidden_actual]
        
        for i, h_dim in enumerate(hidden_dims_list):
            mlp_layers_list.append(nn.Linear(current_feature_dim, h_dim))
            if i in norm_indices:
                mlp_layers_list.append(norm_cls(h_dim))
            if activation_cls:
                mlp_layers_list.append(activation_cls())
            if dropout_p > 0:
                mlp_layers_list.append(nn.Dropout(dropout_p))
            current_feature_dim = h_dim
        
        self.feature_extractor = nn.Sequential(*mlp_layers_list)
        
        # Build the final predictor head
        predictor_head_layers = [nn.Linear(current_feature_dim, output_dim)]
        if final_norm and norm_cls:
            predictor_head_layers.append(norm_cls(output_dim))
        if final_activation_cls:
            predictor_head_layers.append(final_activation_cls())
        if final_dropout and dropout_p > 0:
            predictor_head_layers.append(nn.Dropout(dropout_p))

        self.predictor_head = nn.Sequential(*predictor_head_layers)

        if custom_init:
            self._custom_snn_init(self.feature_extractor)
            self._custom_snn_init(self.predictor_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm_layer:
            x = self.input_norm_layer(x)
        features = self.feature_extractor(x)
        output = self.predictor_head(features)
        return output

    def _custom_snn_init(self, module_block: nn.Sequential):
        """Custom weight initialization (matches original 'snn_init' logic)."""
        for layer in module_block:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias, -bound, bound)


# Helper to parse legacy kwargs and create an MLP instance
def _create_mlp_from_legacy_kwargs(legacy_kwargs: Dict[str, Any],
                                   explicit_input_dim: Optional[int] = None,
                                   explicit_output_dim: Optional[int] = None
                                   ) -> MLP:
    params = {}
    
    params['input_dim'] = explicit_input_dim if explicit_input_dim is not None else legacy_kwargs.get('input_dim')
    params['output_dim'] = explicit_output_dim if explicit_output_dim is not None else legacy_kwargs.get('output_dim')

    if params['input_dim'] is None or params['output_dim'] is None:
        raise ValueError("input_dim and output_dim must be provided either explicitly or in legacy_kwargs")

    hidden_dim_scalar = legacy_kwargs.get('hidden_dim')
    n_hidden = legacy_kwargs.get('n_hidden_layers')
    if isinstance(hidden_dim_scalar, int) and isinstance(n_hidden, int) and n_hidden > 0:
        params['hidden_dims'] = [hidden_dim_scalar] * n_hidden
    elif isinstance(legacy_kwargs.get('hidden_dims'), list):
         params['hidden_dims'] = legacy_kwargs.get('hidden_dims')
    else:
        params['hidden_dims'] = []

    params['activation_cls_str'] = legacy_kwargs.get('activation', "nn.ReLU")
    params['norm_cls_str'] = legacy_kwargs.get('norm_fn') 
    params['apply_norm_to_layers'] = legacy_kwargs.get('norm_layer', "all")
    params['input_norm'] = legacy_kwargs.get('input_norm', False)
    params['dropout_p'] = legacy_kwargs.get('dropout', 0.0) 
    params['final_activation_cls_str'] = legacy_kwargs.get('final_activation')
    params['final_norm'] = legacy_kwargs.get('final_norm', False)
    params['final_dropout'] = legacy_kwargs.get('final_dropout', False)
    params['custom_init'] = legacy_kwargs.get('snn_init', False)
    
    return MLP(**params)

class TaskSpecificMLP(nn.Module):
    def __init__(self,
                 shared_features_dim: int,
                 covariates_dim: int,
                 task_final_output_dim: int,
                 skip_connection_mlp_legacy_kwargs: Dict[str, Any],
                 predictor_mlp_legacy_kwargs: Dict[str, Any]
                ):
        super().__init__()
        
        # Create skip_connection MLP using its legacy kwargs
        # The 'input_dim' is covariates_dim.
        # The 'output_dim' for skip_mlp must be in skip_connection_mlp_legacy_kwargs.
        if 'output_dim' not in skip_connection_mlp_legacy_kwargs:
            raise ValueError("output_dim must be specified in skip_connection_mlp_legacy_kwargs")
        
        self.skip_connection_mlp = _create_mlp_from_legacy_kwargs(
            legacy_kwargs=skip_connection_mlp_legacy_kwargs,
            explicit_input_dim=covariates_dim
        )
        # Get the actual output dimension of the skip MLP
        skip_mlp_actual_output_dim = skip_connection_mlp_legacy_kwargs['output_dim']

        # Create predictor MLP using its legacy kwargs
        # Its input_dim is calculated, its output_dim is task_final_output_dim.
        concatenated_features_dim = shared_features_dim + skip_mlp_actual_output_dim
        
        self.predictor_mlp = _create_mlp_from_legacy_kwargs(
            legacy_kwargs=predictor_mlp_legacy_kwargs,
            explicit_input_dim=concatenated_features_dim,
            explicit_output_dim=task_final_output_dim
        )

    def forward(self, features: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        skip_fts = self.skip_connection_mlp(covariates)
        combined_features = torch.cat((features, skip_fts), dim=-1)
        out = self.predictor_mlp(combined_features)
        return out
    

class OmicsNet(nn.Module):
    def __init__(self, 
                 omics_input_dim: int,
                 outcomes_list: List[str],
                 shared_mlp_kwargs: Dict[str, Any],
                 skip_connection_mlp_kwargs_default: Dict[str, Any], 
                 predictor_mlp_kwargs_default: Dict[str, Any],
                 outcome_specific_mlp_kwargs: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
                 ):
        super().__init__()
        
        # 1. Instantiate Shared MLP
        # The input_dim for shared_mlp is omics_input_dim.
        # Its output_dim must be specified in shared_mlp_kwargs.
        if 'output_dim' not in shared_mlp_kwargs:
            raise ValueError("output_dim must be specified in shared_mlp_kwargs")

        self.shared_mlp = _create_mlp_from_legacy_kwargs(
            legacy_kwargs=shared_mlp_kwargs,
            explicit_input_dim=omics_input_dim
        )
        shared_features_actual_dim = shared_mlp_kwargs['output_dim']

        # 2. Instantiate Task-Specific Heads
        self.output_layers = nn.ModuleDict()
        if outcome_specific_mlp_kwargs is None:
            outcome_specific_mlp_kwargs = {}

        for outcome_name in outcomes_list:
            # Get task-specific kwargs or use defaults
            task_skip_kwargs = outcome_specific_mlp_kwargs.get(outcome_name, {}).get(
                "skip_connection_mlp_kwargs", skip_connection_mlp_kwargs_default.copy()
            )
            task_predictor_kwargs = outcome_specific_mlp_kwargs.get(outcome_name, {}).get(
                "predictor_mlp_kwargs", predictor_mlp_kwargs_default.copy()
            )
            
            # The final output dimension for this task's predictor head.
            # This should be defined in predictor_mlp_kwargs_default or overridden by task_predictor_kwargs.
            if 'output_dim' not in task_predictor_kwargs:
                task_final_output_dim = 1 # Defaulting to 1 if not found; make this configurable
                print(f"Warning: output_dim not found for predictor_mlp for task '{outcome_name}'. Defaulting to {task_final_output_dim}.")
            else:
                task_final_output_dim = task_predictor_kwargs['output_dim']


            self.output_layers[outcome_name] = TaskSpecificMLP(
                shared_features_dim=shared_features_actual_dim,
                covariates_dim=omics_input_dim, 
                task_final_output_dim=task_final_output_dim, 
                skip_connection_mlp_legacy_kwargs=task_skip_kwargs,
                predictor_mlp_legacy_kwargs=task_predictor_kwargs
            )
        
    def forward(self, omics_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        shared_features = self.shared_mlp(omics_data)
        for outcome_name, task_head_mlp in self.output_layers.items():
            outputs[outcome_name] = task_head_mlp(features=shared_features, covariates=omics_data)
        return outputs