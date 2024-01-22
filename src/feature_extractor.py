from src.transformer_feature_extractor import TransformerFeatureExtractor
from src.convolution_feature_extractor import ConvolutionFeatureExtractor


def _initialize_transformer_feature_extractor(feature_extractor_config):
    feature_extractor_class = TransformerFeatureExtractor
    return feature_extractor_class, feature_extractor_config


def _initialize_convolution_feature_extractor(feature_extractor_config):
    feature_extractor_class = ConvolutionFeatureExtractor
    feature_extractor_kwargs = feature_extractor_config["feature_extractor"].copy()
    feature_extractor_kwargs.pop("name", None)
    return feature_extractor_class, feature_extractor_config


def feature_extractor_initializer(feature_extractor_config):
    config = feature_extractor_config.copy()
    feature_extractor_type = config.pop("type")
    if feature_extractor_type == "Transformer":
        return TransformerFeatureExtractor, config
    elif feature_extractor_type == "Convolution":
        return ConvolutionFeatureExtractor, config
    else:
        raise TypeError("Unexpected feature extractor type : ", feature_extractor_type)
