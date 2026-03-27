# API Reference

Complete API documentation for the Human-Aligned Phishing Explanation System.

## Table of Contents

- [Data Structures](#data-structures)
- [Generators](#generators)
- [Component Analyzers](#component-analyzers)
- [Explainers](#explainers)
- [Metrics](#metrics)
- [Utilities](#utilities)

---

## Data Structures

### EmailData

```python
@dataclass
class EmailData:
    """Complete email data for explanation."""
    sender: EmailAddress
    recipients: List[EmailAddress]
    subject: str
    body: str
    urls: List[URL] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    timestamp: Optional[str] = None
    email_id: Optional[str] = None
    category: EmailCategory = EmailCategory.SAFE
```

**Example Usage**:
```python
email = EmailData(
    sender=EmailAddress(
        display_name="Netflix Support",
        email="support@netfliix-security.com"
    ),
    recipients=[],
    subject="URGENT: Your account will be suspended",
    body="Your account will be suspended..."
)
```

### EmailAddress

```python
@dataclass
class EmailAddress:
    """Email address with display name."""
    display_name: Optional[str]
    email: str
    is_suspicious: bool = False
    suspicion_reasons: List[str] = field(default_factory=list)
```

### URL

```python
@dataclass
class URL:
    """URL with safety information."""
    original: str
    domain: str
    path: Optional[str] = None
    has_https: bool = False
    domain_age_days: Optional[int] = None
    is_suspicious: bool = False
    suspicion_reasons: List[str] = field(default_factory=list)
```

### Attachment

```python
@dataclass
class Attachment:
    """Email attachment with risk information."""
    filename: str
    file_type: str
    size_bytes: int
    has_macros: bool = False
    is_dangerous: bool = False
    risk_reasons: List[str] = field(default_factory=list)
```

### ModelOutput

```python
@dataclass
class ModelOutput:
    """Model prediction output."""
    predicted_label: EmailCategory
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
```

### Explanation

```python
@dataclass
class Explanation:
    """Complete explanation for an email prediction."""
    email: EmailData
    model_prediction: ModelOutput

    # Component explanations (cognitive order)
    sender_explanation: Optional[SenderExplanation] = None
    subject_explanation: Optional[SubjectExplanation] = None
    body_explanation: Optional[BodyExplanation] = None
    url_explanation: Optional[URLExplanation] = None
    attachment_explanation: Optional[AttachmentExplanation] = None

    # Advanced explanations
    feature_importance: Optional[FeatureImportance] = None
    attention_visualization: Optional[AttentionVisualization] = None
    counterfactuals: List[CounterfactualExplanation] = field(default_factory=list)
    comparative: Optional[ComparativeExplanation] = None

    # Metadata
    explanation_types: List[ExplanationType] = field(default_factory=list)
    generation_time_ms: float = 0.0
    is_federated: bool = False
```

---

## Generators

### HumanAlignedGenerator

Main generator for human-aligned explanations.

```python
class HumanAlignedGenerator(BaseExplanationGenerator):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_feature_importance: bool = True,
        use_attention: bool = True,
        use_counterfactuals: bool = True,
        use_comparisons: bool = True
    )
```

**Methods**:

#### `generate_explanation()`
```python
def generate_explanation(
    self,
    email: EmailData,
    model_prediction: ModelOutput,
    attention_weights: Optional[Any] = None,
    **kwargs
) -> Explanation
```
Generate comprehensive explanation in cognitive order.

**Parameters**:
- `email`: Email to explain
- `model_prediction`: Model's prediction output
- `attention_weights`: Optional pre-computed attention weights

**Returns**:
- `Explanation` object

**Example**:
```python
generator = HumanAlignedGenerator()
explanation = generator.generate_explanation(email, prediction)
```

#### `generate_with_timing()`
```python
def generate_with_timing(
    self,
    email: EmailData,
    model_prediction: ModelOutput,
    **kwargs
) -> Explanation
```
Generate explanation and track generation time.

#### `generate_batch()`
```python
def generate_batch(
    self,
    emails: List[EmailData],
    model_predictions: List[ModelOutput]
) -> List[Explanation]
```
Generate explanations for multiple emails.

### FederatedExplanationGenerator

Privacy-preserving generator for federated learning.

```python
class FederatedExplanationGenerator(HumanAlignedGenerator):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        local_statistics: Optional[Dict[str, Any]] = None,
        privacy_budget: float = 1.0
    )
```

**Methods**:

#### `generate_local_explanation()`
```python
def generate_local_explanation(
    self,
    email: EmailData,
    model_prediction: ModelOutput,
    use_global_features: bool = False
) -> Explanation
```
Generate privacy-preserving local explanation.

#### `generate_with_differential_privacy()`
```python
def generate_with_differential_privacy(
    self,
    email: EmailData,
    model_prediction: ModelOutput,
    epsilon: Optional[float] = None
) -> Explanation
```
Generate explanation with differential privacy noise.

---

## Component Analyzers

### SenderAnalyzer

```python
class SenderAnalyzer:
    def analyze(self, email: EmailData) -> SenderExplanation
```

**Returns**: `SenderExplanation`
- `is_suspicious`: bool
- `confidence`: float
- `reasons`: List[str]
- `domain_reputation`: str ("good", "unknown", "poor")
- `display_name_mismatch`: bool
- `lookalike_domain`: bool

### SubjectAnalyzer

```python
class SubjectAnalyzer:
    def analyze(self, email: EmailData) -> SubjectExplanation
```

**Returns**: `SubjectExplanation`
- `is_suspicious`: bool
- `confidence`: float
- `reasons`: List[str]
- `urgency_keywords`: List[str]
- `unusual_formatting`: List[str]

### BodyAnalyzer

```python
class BodyAnalyzer:
    def analyze(self, email: EmailData) -> BodyExplanation
```

**Returns**: `BodyExplanation`
- `is_suspicious`: bool
- `confidence`: float
- `reasons`: List[str]
- `social_engineering_tactics`: List[str]
- `grammar_issues`: List[str]
- `pressure_language`: List[str]

### URLAnalyzer

```python
class URLAnalyzer:
    def analyze(self, email: EmailData) -> URLExplanation
```

**Returns**: `URLExplanation`
- `is_suspicious`: bool
- `confidence`: float
- `reasons`: List[str]
- `suspicious_urls`: List[Dict[str, Any]]
- `safe_urls`: List[str]

### AttachmentAnalyzer

```python
class AttachmentAnalyzer:
    def analyze(self, email: EmailData) -> AttachmentExplanation
```

**Returns**: `AttachmentExplanation`
- `is_suspicious`: bool
- `confidence`: float
- `reasons`: List[str]
- `dangerous_attachments`: List[Dict[str, Any]]

---

## Explainers

### FeatureBasedExplainer

```python
class FeatureBasedExplainer:
    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        use_kernel_shap: bool = True
    )

    def explain(
        self,
        email: EmailData,
        feature_names: Optional[List[str]] = None
    ) -> FeatureImportance
```

**SimpleFeatureExplainer** (no SHAP dependency):
```python
from src.explainers.feature_based import SimpleFeatureExplainer

explainer = SimpleFeatureExplainer()
feature_importance = explainer.explain(email, model_prediction)
```

### AttentionBasedExplainer

```python
class AttentionBasedExplainer:
    def explain(
        self,
        email: EmailData,
        max_length: int = 128
    ) -> AttentionVisualization
```

**SimpleAttentionExplainer** (no transformer dependency):
```python
from src.explainers.attention_based import SimpleAttentionExplainer

explainer = SimpleAttentionExplainer()
attention_viz = explainer.explain(email)
```

### CounterfactualExplainer

```python
class CounterfactualExplainer:
    def generate_counterfactuals(
        self,
        email: EmailData,
        original_prediction: ModelOutput,
        num_cf: int = 3
    ) -> List[CounterfactualExplanation]
```

**SimpleCounterfactualExplainer** (no model dependency):
```python
from src.explainers.counterfactual import SimpleCounterfactualExplainer

explainer = SimpleCounterfactualExplainer()
counterfactuals = explainer.generate_counterfactuals(email, prediction)
```

### ComparativeExplainer

```python
class ComparativeExplainer:
    def explain(self, email: EmailData) -> ComparativeExplanation
```

---

## Metrics

### Faithfulness

```python
from src.metrics.faithfulness import compute_faithfulness

faithfulness_score = compute_faithfulness(
    explanation=explanation,
    model=model,
    email=email,
    num_perturbations=100
)
# Returns: float (0.0 to 1.0)
```

### Consistency

```python
from src.metrics.consistency import compute_consistency

consistency_score = compute_consistency(
    generator=generator,
    emails=emails,
    predictions=predictions,
    threshold=0.8
)
# Returns: float (0.0 to 1.0)
```

### Human Evaluation

```python
from src.metrics.human_eval import HumanEvaluationMetrics

metrics = HumanEvaluationMetrics()
task = metrics.create_evaluation_task(email_data, explanation, ground_truth)
metrics.record_result(result)
report = metrics.generate_report()
```

---

## Utilities

### Text Processing

```python
from src.utils.text_processing import (
    extract_urls,
    extract_email_addresses,
    normalize_text,
    detect_urgency_keywords,
    detect_pressure_language
)

# Extract URLs from text
urls = extract_urls(text)

# Extract email addresses
emails = extract_email_addresses(text)

# Normalize text
clean = normalize_text(text)

# Detect urgency keywords
urgency = detect_urgency_keywords(subject)
```

### Formatters

```python
from src.utils.formatters import (
    format_explanation_for_user,
    format_explanation_for_analyst,
    format_confidence_score,
    get_actionable_advice
)

# Format for end users
user_friendly = format_explanation_for_user(explanation)

# Format for analysts
technical = format_explanation_for_analyst(explanation)

# Get confidence description
conf_desc = format_confidence_score(0.92)  # "very high"

# Get actionable advice
advice = get_actionable_advice(explanation)
```

---

## Streamlit Apps

### User App

```bash
streamlit run src/ui/user_app.py
```

### Analyst Interface

```bash
streamlit run src/ui/analyst_interface.py
```

---

## Error Handling

### Custom Exceptions

```python
# Input validation error
try:
    generator.validate_input(email, prediction)
except ValueError as e:
    print(f"Invalid input: {e}")

# Explanation generation error
if explanation.generation_time_ms > 500:
    print("Warning: Generation time exceeds 500ms target")
```

---

## Type Hints

All functions use Python type hints. Example:

```python
from typing import List, Dict, Optional, Any

def generate_explanation(
    email: EmailData,
    model_prediction: ModelOutput,
    **kwargs: Any
) -> Explanation:
    ...
```

---

For more examples, see `notebooks/explanation_examples.ipynb`.
