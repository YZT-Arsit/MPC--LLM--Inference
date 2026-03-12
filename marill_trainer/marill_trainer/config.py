import enum
import re

class LayerConfig(enum.Enum):
    Full = 1
    BottomFreezing = 2
    SpacedFreezing = 3
    Pruning = 4

    def __init__(self, num_layers = None):
        self._num_layers = None

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        if self in [LayerConfig.BottomFreezing, LayerConfig.SpacedFreezing, LayerConfig.Pruning]:
            self._num_layers = value
        else:
            raise ValueError("num_layers can not be set when all layers are fine-tuned")

    def __str__(self):
        if self is LayerConfig.BottomFreezing:
            return f"lf{self._num_layers}"
        elif self is LayerConfig.SpacedFreezing:
            return f"ls{self._num_layers}"
        elif self is LayerConfig.Pruning:
            return f"lp{self._num_layers}"
        elif self is LayerConfig.Full:
            return ""
        else:
            raise ValueError("Invalid LayerConfig")

    def from_str(string):
        match = re.match(r'(\w+)(?:\{(\d+)\})?$', string)
        if match:
            operation = match.group(1)
            value = int(match.group(2)) if match.group(2) is not None else None
            if operation == "full":
                assert(value is None)
                config = LayerConfig.Full
            elif operation == "bottom":
                config = LayerConfig.BottomFreezing
            elif operation == "spaced":
                config = LayerConfig.SpacedFreezing
            elif operation == "prune":
                config = LayerConfig.Pruning
            else:
                raise ValueError(f"Invalid LayerConfig: {string}")
            if value is not None:
                config.num_layers = value
            return config
        else:
            raise ValueError(f"Invalid LayerConfig: {string}")

    def as_dict(self):
        config = {}
        config['type'] = str(self.name)
        if self.num_layers is not None:
            config['num_layers'] = self.num_layers
        return config

class WeightConfig(enum.Enum):
    Full = 1
    LoRA = 2

    def __init__(self, rank=None):
        self._rank = None

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        if self in [WeightConfig.LoRA]:
            self._rank = value
        else:
            raise ValueError("rank can not be set when all weights are fine-tuned")

    def __str__(self):
        if self is WeightConfig.LoRA:
            return f"lora{self._rank}"
        elif self is WeightConfig.Full:
            return ""
        else:
            raise ValueError("Invalid LinearConfig")

    def from_str(string):
        match = re.match(r'(\w+)(?:\{(\d+)\})?$', string)
        if match:
            operation = match.group(1)
            value = int(match.group(2)) if match.group(2) is not None else None
            if operation == "full":
                assert(value is None)
                config = WeightConfig.Full
            elif operation == "lora":
                config = WeightConfig.LoRA
            else:
                raise ValueError(f"Invalid WeightConfig: {string}")
            if value is not None:
                config.rank = value
            return config
        else:
            raise ValueError(f"Invalid WeightConfig: {string}")

    def as_dict(self):
        config = {}
        config['type'] = str(self.name)
        if self.rank is not None:
            config['rank'] = self.rank
        return config

class HeadConfig(enum.Enum):
    Default = 1
    Merging = 2
    PermutedMerging = 3
    Clustering = 4
    EvenClustering = 5
    Pruning = 6
    UniformPruning = 7

    def __init__(self, factor=None):
        self._factor = None

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        if self in [HeadConfig.Merging, HeadConfig.PermutedMerging, HeadConfig.Clustering, HeadConfig.EvenClustering]:
            self._factor = value
        elif self in [HeadConfig.Pruning, HeadConfig.UniformPruning]:
            assert(value > 1.0)
            self._factor = value
        else:
            raise ValueError("factor can not be set when using default head config")

    def __str__(self):
        if self is HeadConfig.Merging:
            return f"hm{self._factor}"
        elif self is HeadConfig.PermutedMerging:
            return f"phm{self._factor}"
        elif self is HeadConfig.Clustering:
            return f"hc{self._factor}"
        elif self is HeadConfig.EvenClustering:
            return f"ehc{self._factor}"
        elif self is HeadConfig.Pruning:
            return f"hp{self._factor}"
        elif self is HeadConfig.UniformPruning:
            return f"uhp{self._factor}"
        elif self is HeadConfig.Default:
            return ""
        else:
            raise ValueError("Invalid HeadConfig")

    def from_str(string):
        match = re.match(r'(\w+)(?:\{(\d+)\})?$', string)
        if match:
            operation = match.group(1)
            value = int(match.group(2)) if match.group(2) is not None else None
            if operation == "default":
                assert(value is None)
                config = HeadConfig.Default
            elif operation == "merge":
                config = HeadConfig.Merging
            elif operation == "permuted_merge":
                config = HeadConfig.PermutedMerging
            elif operation == "cluster":
                config = HeadConfig.Clustering
            elif operation == "even_cluster":
                config = HeadConfig.EvenClustering
            elif operation == "prune":
                config = HeadConfig.Pruning
            elif operation == "uniform_prune":
                config = HeadConfig.UniformPruning
            else:
                raise ValueError(f"Invalid HeadConfig: {string}")
            if value is not None:
                config.factor = value
            return config
        else:
            raise ValueError(f"Invalid HeadConfig: {string}")

    def as_dict(self):
        config = {}
        config['type'] = str(self.name)
        if self.factor is not None:
            config['factor'] = self.factor
        return config

class ActType(enum.Enum):
    Default = 0
    Quad = 1
    ReLU = 2

    def __str__(self):
        if self is ActType.Default:
            return f"silu"
        elif self is ActType.Quad:
            return f"quad"
        elif self is ActType.ReLU:
            return f"relu"
        else:
            raise ValueError("Invalid ActType")

    def from_str(string):
        if string == "default":
            return ActType.Default
        elif string == "silu":
            return ActType.Default
        elif string == "quad":
            return ActType.Quad
        elif string == "relu":
            return ActType.ReLU
        else:
            raise ValueError(f"Invalid ActType: {string}")

class SoftmaxType(enum.Enum):
    Default = 0
    TwoQuad = 1
    LearnableTwoQuad = 2
    TwoReLU = 3
    Scaling = 4

    def __str__(self):
        if self is SoftmaxType.Default:
            return f"smax"
        elif self is SoftmaxType.TwoQuad:
            return f"2quad"
        elif self is SoftmaxType.LearnableTwoQuad:
            return f"l2quad"
        elif self is SoftmaxType.TwoReLU:
            return f"2relu"
        elif self is SoftmaxType.Scaling:
            return f"scale"
        else:
            raise ValueError("Invalid SoftmaxType")

    def from_str(string):
        if string == "default":
            return SoftmaxType.Default
        elif string == "smax":
            return SoftmaxType.Default
        elif string == "2quad":
            return SoftmaxType.TwoQuad
        elif string == "l2quad":
            return SoftmaxType.LearnableTwoQuad
        elif string == "2relu":
            return SoftmaxType.TwoReLU
        elif string == "scale":
            return SoftmaxType.Scaling
        else:
            raise ValueError(f"Invalid SoftmaxType: {string}")