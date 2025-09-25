import os
import json
import torch
from typing import List, Optional

from .base import BaseProbe


class ProbeSet:
    """A collection of probes, typically acting on the same activation space."""

    def __init__(self, probes: List[BaseProbe]):
        self.probes = probes

        # Validate compatibility and store common metadata
        if probes:
            first_probe = probes[0]
            self.input_dim = getattr(first_probe.config, "input_size", None)
            self.model_name = getattr(first_probe.config, "model_name", None)
            self.hook_point = getattr(first_probe.config, "hook_point", None)
            self.hook_layer = getattr(first_probe.config, "hook_layer", None)
            self.device = getattr(first_probe.config, "device", None)
            self.dtype = getattr(
                first_probe, "dtype", torch.float32
            )  # Get dtype from probe instance

            if self.input_dim is None:
                raise ValueError(
                    f"First probe '{first_probe.name}' must have input_size defined in its config."
                )

            for i, p in enumerate(probes[1:], 1):
                p_dim = getattr(p.config, "input_size", None)
                if p_dim != self.input_dim:
                    raise ValueError(
                        f"Probe {i} ('{p.name}') has input dimension {p_dim}, but expected {self.input_dim} based on the first probe."
                    )
                # Optional: Stricter checks for model_name, hook_point etc.
                p_model = getattr(p.config, "model_name", None)
                if p_model != self.model_name:
                    print(
                        f"Warning: Probe {i} ('{p.name}') model '{p_model}' differs from set model '{self.model_name}'."
                    )
                p_hook = getattr(p.config, "hook_point", None)
                if p_hook != self.hook_point:
                    print(
                        f"Warning: Probe {i} ('{p.name}') hook point '{p_hook}' differs from set hook point '{self.hook_point}'."
                    )

        else:
            self.input_dim = None
            self.model_name = None
            self.hook_point = None
            self.hook_layer = None
            self.device = None
            self.dtype = torch.float32  # Default dtype if empty set

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot products with all probes' normalized directions.

        Args:
            acts: Activations to project, shape [..., d_model]

        Returns:
            Projected values, shape [..., n_probes]
        """
        if not self.probes:
            return torch.empty(
                acts.shape[:-1] + (0,), device=acts.device, dtype=self.dtype
            )

        # Ensure activations are on the correct device and dtype
        target_device = self.device or acts.device
        acts = acts.to(device=target_device, dtype=self.dtype)

        # Get normalized directions for all probes
        # Ensure all directions are on the target device and correct dtype
        try:
            weight_matrix = torch.stack(
                [
                    p.get_direction(normalized=True).to(
                        device=target_device, dtype=self.dtype
                    )
                    for p in self.probes
                ]
            )  # Shape [n_probes, d_model]
        except Exception as e:
            probe_shapes = [
                (p.name, p.get_direction(normalized=True).shape) for p in self.probes
            ]
            raise RuntimeError(
                f"Error stacking probe directions during encode. Check probe consistency. Shapes: {probe_shapes}. Original error: {e}"
            )

        # Check for dimension mismatch before einsum
        if acts.shape[-1] != weight_matrix.shape[-1]:
            raise ValueError(
                f"Activation dimension ({acts.shape[-1]}) does not match probe direction dimension ({weight_matrix.shape[-1]})."
            )

        # Project all at once using einsum for flexibility with batch dimensions
        return torch.einsum("...d,nd->...n", acts, weight_matrix)

    def __getitem__(self, idx) -> BaseProbe:
        """Get a probe by index."""
        return self.probes[idx]

    def __len__(self) -> int:
        """Get number of probes."""
        return len(self.probes)

    def save(self, directory: str, use_json: bool = False) -> None:
        """Save all probes to a directory, including an index file.

        Args:
            directory: Directory to save the probes.
            use_json: If True, save probes in JSON format, otherwise use .pt.
        """
        os.makedirs(directory, exist_ok=True)

        # Save index file with common metadata and list of probe files
        index = {
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "hook_layer": self.hook_layer,
            "format": "json" if use_json else "pt",
            "probes": [],
        }

        # Save each probe individually
        for i, probe in enumerate(self.probes):
            # Sanitize probe name for filename (replace non-alphanumeric with underscore)
            safe_name = "".join(c if c.isalnum() else "_" for c in probe.name)
            # Ensure filename is not excessively long
            max_len = 60  # Max length for the name part
            safe_name = safe_name[:max_len]

            filename = f"probe_{i}_{safe_name}.{'json' if use_json else 'pt'}"
            filepath = os.path.join(directory, filename)

            try:
                if use_json:
                    probe.save_json(filepath)
                else:
                    probe.save(filepath)
            except Exception as e:
                print(f"Error saving probe {i} ('{probe.name}') to {filepath}: {e}")
                # Continue saving other probes
                continue

            # Add entry to index
            index["probes"].append(
                {
                    "name": probe.name,
                    "file": filename,
                    "probe_type": probe.__class__.__name__,
                }
            )

        # Save the index file
        index_path = os.path.join(directory, "index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            print(f"Error saving index file to {index_path}: {e}")

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "ProbeSet":
        """Load a ProbeSet from a directory containing probes and an index.json file.

        Args:
            directory: Directory containing the probes and index.json.
            device: Optional device override for loading probes. If None, uses device
                    specified during saving (from index/probe metadata) or default.

        Returns:
            ProbeSet instance
        """
        index_path = os.path.join(directory, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found in directory: {directory}")

        # Load index
        try:
            with open(index_path) as f:
                index = json.load(f)
        except Exception as e:
            raise IOError(f"Error reading index file {index_path}: {e}")

        # Load each probe listed in the index
        probes = []
        save_format = index.get(
            "format", "pt"
        )  # Default to .pt if format not specified

        for i, entry in enumerate(index.get("probes", [])):
            filename = entry.get("file")
            probe_type_name = entry.get("probe_type")
            probe_name = entry.get("name", f"probe_{i}")

            if not filename or not probe_type_name:
                print(
                    f"Warning: Skipping entry {i} in index due to missing 'file' or 'probe_type': {entry}"
                )
                continue

            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(
                    f"Warning: File {filepath} listed in index for probe '{probe_name}' not found. Skipping probe."
                )
                continue

            # Dynamically get the probe class (delegated to BaseProbe.load/.load_json)
            # We need a starting point - BaseProbe itself
            try:
                if save_format == "json":
                    # BaseProbe.load_json will determine the correct class from metadata
                    probe = BaseProbe.load_json(filepath, device=device)
                else:
                    # BaseProbe.load will determine the correct class from saved .pt file
                    probe = BaseProbe.load(filepath, device=device)

                # Optional: Verify loaded probe type matches index entry
                if probe.__class__.__name__ != probe_type_name:
                    print(
                        f"Warning: Loaded probe type '{probe.__class__.__name__}' from {filename} does not match index type '{probe_type_name}'."
                    )

                probes.append(probe)

            except Exception as e:
                print(
                    f"Error loading probe '{probe_name}' from {filepath}: {e}. Skipping probe."
                )
                # Decide whether to raise an error or just skip the problematic probe
                # For now, skipping
                continue

        if not probes:
            print("Warning: Loaded an empty ProbeSet.")

        # Create the ProbeSet instance
        return cls(probes)
