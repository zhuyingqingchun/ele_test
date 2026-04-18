from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateConfig:
    name: str
    include_label_tokens: bool
    include_family_tokens: bool
    include_location_tokens: bool
    include_boundary_tokens: bool
    include_mechanism: bool
    include_signal_priors: bool
    include_contrast: bool
    include_condition: bool
    include_severity_bucket: bool
    use_compact_phrases: bool
    use_full_sentences: bool
    shuffle_pieces: bool
    max_mechanism_items: int
    max_signal_items: int
    max_contrast_items: int


TEMPLATE_CONFIGS: dict[str, TemplateConfig] = {
    # 旧式强模板：不推荐，只用于对照实验
    "label_rich": TemplateConfig(
        name="label_rich",
        include_label_tokens=True,
        include_family_tokens=True,
        include_location_tokens=True,
        include_boundary_tokens=True,
        include_mechanism=True,
        include_signal_priors=True,
        include_contrast=False,
        include_condition=False,
        include_severity_bucket=False,
        use_compact_phrases=False,
        use_full_sentences=True,
        shuffle_pieces=False,
        max_mechanism_items=1,
        max_signal_items=99,
        max_contrast_items=0,
    ),
    # 推荐：弱模板、强特征
    "feature_only": TemplateConfig(
        name="feature_only",
        include_label_tokens=False,
        include_family_tokens=False,
        include_location_tokens=False,
        include_boundary_tokens=False,
        include_mechanism=True,
        include_signal_priors=True,
        include_contrast=True,
        include_condition=False,
        include_severity_bucket=False,
        use_compact_phrases=True,
        use_full_sentences=False,
        shuffle_pieces=True,
        max_mechanism_items=2,
        max_signal_items=4,
        max_contrast_items=2,
    ),
    # 更偏机理表述
    "mechanism_focus": TemplateConfig(
        name="mechanism_focus",
        include_label_tokens=False,
        include_family_tokens=False,
        include_location_tokens=False,
        include_boundary_tokens=False,
        include_mechanism=True,
        include_signal_priors=True,
        include_contrast=True,
        include_condition=False,
        include_severity_bucket=False,
        use_compact_phrases=False,
        use_full_sentences=True,
        shuffle_pieces=True,
        max_mechanism_items=2,
        max_signal_items=3,
        max_contrast_items=1,
    ),
    # 增加强工况语义
    "feature_conditioned": TemplateConfig(
        name="feature_conditioned",
        include_label_tokens=False,
        include_family_tokens=False,
        include_location_tokens=False,
        include_boundary_tokens=False,
        include_mechanism=True,
        include_signal_priors=True,
        include_contrast=True,
        include_condition=True,
        include_severity_bucket=True,
        use_compact_phrases=True,
        use_full_sentences=False,
        shuffle_pieces=True,
        max_mechanism_items=2,
        max_signal_items=4,
        max_contrast_items=2,
    ),
    # 极简原型文本
    "prototype_minimal": TemplateConfig(
        name="prototype_minimal",
        include_label_tokens=False,
        include_family_tokens=False,
        include_location_tokens=False,
        include_boundary_tokens=False,
        include_mechanism=False,
        include_signal_priors=True,
        include_contrast=True,
        include_condition=False,
        include_severity_bucket=False,
        use_compact_phrases=True,
        use_full_sentences=False,
        shuffle_pieces=True,
        max_mechanism_items=0,
        max_signal_items=3,
        max_contrast_items=1,
    ),
    "fine_label_feature": TemplateConfig(
    name="fine_label_feature",
    include_label_tokens=True,
    include_family_tokens=False,
    include_location_tokens=False,
    include_boundary_tokens=False,
    include_mechanism=True,
    include_signal_priors=True,
    include_contrast=True,
    include_condition=False,
    include_severity_bucket=False,
    use_compact_phrases=True,
    use_full_sentences=False,
    shuffle_pieces=True,
    max_mechanism_items=2,
    max_signal_items=4,
    max_contrast_items=2,
),
}


DEFAULT_TEMPLATE_NAME = "fine_label_feature"