"""
Human evaluation metrics and user study design.

Provides protocols and metrics for evaluating explanation quality
with human participants.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class HumanEvaluationResult:
    """Result from human evaluation."""
    participant_id: str
    explanation_id: str
    understandability: float  # 1-5 Likert scale
    helpfulness: float  # 1-5 Likert scale
    trust: float  # 1-5 Likert scale
    actionability: float  # 1-5 Likert scale
    time_to_understand: float  # seconds
    correct_decision: bool  # Did they make right decision?
    confidence_in_decision: float  # 1-5
    feedback: str  # Qualitative feedback


class HumanEvaluationMetrics:
    """
    Metrics for human evaluation of explanations.

    Based on XAI evaluation frameworks and user study protocols.
    """

    # Likert scale questions
    UNDERSTANDABILITY_QUESTIONS = [
        "I understood why the email was flagged",
        "The explanation was clear and easy to follow",
        "The technical language was understandable",
        "I could explain this to another person"
    ]

    HELPFULNESS_QUESTIONS = [
        "The explanation helped me make a decision",
        "I learned something about phishing from this explanation",
        "The explanation addressed my concerns",
        "The information was relevant"
    ]

    TRUST_QUESTIONS = [
        "I trust the explanation provided",
        "The explanation seems accurate",
        "I believe the system is reliable",
        "I would use this system again"
    ]

    ACTIONABILITY_QUESTIONS = [
        "I know what action to take based on this explanation",
        "The explanation provides clear next steps",
        "I can apply this knowledge to future emails"
    ]

    def __init__(self):
        """Initialize human evaluation metrics."""
        self.results: List[HumanEvaluationResult] = []

    def create_evaluation_task(
        self,
        email_data: Dict[str, Any],
        explanation: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Create an evaluation task for user study.

        Args:
            email_data: Email information
            explanation: Generated explanation
            ground_truth: True label (phishing/safe)

        Returns:
            Task dictionary
        """
        return {
            'task_id': f"task_{len(self.results) + 1}",
            'email': email_data,
            'explanation': explanation,
            'questions': {
                'understandability': self.UNDERSTANDABILITY_QUESTIONS,
                'helpfulness': self.HELPFULNESS_QUESTIONS,
                'trust': self.TRUST_QUESTIONS,
                'actionability': self.ACTIONABILITY_QUESTIONS
            },
            'decision_prompt': f"Based on the explanation, is this email phishing or safe?",
            'ground_truth': ground_truth
        }

    def record_result(self, result: HumanEvaluationResult):
        """Record evaluation result."""
        self.results.append(result)

    def compute_average_scores(self) -> Dict[str, float]:
        """Compute average scores across all evaluations."""
        if not self.results:
            return {}

        return {
            'understandability': sum(r.understandability for r in self.results) / len(self.results),
            'helpfulness': sum(r.helpfulness for r in self.results) / len(self.results),
            'trust': sum(r.trust for r in self.results) / len(self.results),
            'actionability': sum(r.actionability for r in self.results) / len(self.results),
            'avg_time_to_understand': sum(r.time_to_understand for r in self.results) / len(self.results),
            'accuracy': sum(r.correct_decision for r in self.results) / len(self.results),
            'avg_confidence': sum(r.confidence_in_decision for r in self.results) / len(self.results)
        }

    def compute_agreement(self) -> float:
        """
        Compute inter-rater agreement (Krippendorff's alpha simplified).

        Measures how much raters agree on explanations.
        """
        if len(self.results) < 2:
            return 1.0

        # Simplified agreement: proportion of same decisions
        decisions = [r.correct_decision for r in self.results]
        if not decisions:
            return 1.0

        majority = max(set(decisions), key=decisions.count)
        agreement = sum(1 for d in decisions if d == majority) / len(decisions)

        return agreement

    def generate_report(self) -> str:
        """Generate human evaluation report."""
        scores = self.compute_average_scores()

        if not scores:
            return "No evaluation results available."

        report = [
            "# Human Evaluation Report",
            "",
            f"**Total Participants**: {len(self.results)}",
            "",
            "## Average Scores (1-5 Likert Scale)",
            "",
            f"- **Understandability**: {scores.get('understandability', 0):.2f}/5.0",
            f"- **Helpfulness**: {scores.get('helpfulness', 0):.2f}/5.0",
            f"- **Trust**: {scores.get('trust', 0):.2f}/5.0",
            f"- **Actionability**: {scores.get('actionability', 0):.2f}/5.0",
            "",
            "## Performance Metrics",
            "",
            f"- **Accuracy**: {scores.get('accuracy', 0):.2%}",
            f"- **Avg Confidence**: {scores.get('avg_confidence', 0):.2f}/5.0",
            f"- **Avg Time to Understand**: {scores.get('avg_time_to_understand', 0):.1f}s",
            "",
            f"- **Inter-rater Agreement**: {self.compute_agreement():.2%}",
            "",
            "## Qualitative Feedback",
            ""
        ]

        # Add feedback samples
        for i, result in enumerate(self.results[:5], 1):
            if result.feedback:
                report.append(f"### Participant {result.participant_id}")
                report.append(f"> {result.feedback}")
                report.append("")

        return "\n".join(report)


class UserStudyDesign:
    """
    User study protocol for evaluating explanations.

    Follows best practices from HCI/XAI research.
    """

    @staticmethod
    def create_study_protocol() -> Dict[str, Any]:
        """Create user study protocol."""
        return {
            'study_title': 'Evaluation of Human-Aligned Phishing Explanations',
            'duration_minutes': 45,
            'compensation': '$20 gift card',

            'participants': {
                'target_n': 30,
                'inclusion_criteria': [
                    '18 years or older',
                    'Use email regularly',
                    'No prior training in cybersecurity'
                ],
                'exclusion_criteria': [
                    'Professional security analysts',
                    'Previous participants in similar studies'
                ]
            },

            'procedure': [
                {
                    'phase': 'Introduction',
                    'duration_minutes': 5,
                    'description': 'Consent form and study overview'
                },
                {
                    'phase': 'Training',
                    'duration_minutes': 10,
                    'description': 'Example explanations and interface tutorial'
                },
                {
                    'phase': 'Main Task',
                    'duration_minutes': 25,
                    'description': 'Evaluate 10 email explanations'
                },
                {
                    'phase': 'Survey',
                    'duration_minutes': 5,
                    'description': 'Post-study questionnaire and feedback'
                }
            ],

            'measures': {
                'independent_variables': [
                    'Explanation type (feature-based, attention-based, counterfactual, comparative)',
                    'Email type (phishing, safe, suspicious)',
                    'Participant expertise'
                ],
                'dependent_variables': [
                    'Decision accuracy',
                    'Time to decision',
                    'Understandability (1-5)',
                    'Helpfulness (1-5)',
                    'Trust (1-5)',
                    'Actionability (1-5)'
                ]
            },

            'hypotheses': [
                'H1: Explanations following cognitive order (sender→subject→body→URL) will be more understandable than random order',
                'H2: Counterfactual explanations will improve actionability compared to feature-based alone',
                'H3: Non-technical language will improve understandability for non-experts',
                'H4: Explanations will increase trust in phishing detection system'
            ],

            'ethical_considerations': [
                'Informed consent',
                'Right to withdraw at any time',
                'Data anonymization',
                'No sensitive data collected',
                'IRB approval required'
            ]
        }

    @staticmethod
    def create_demographic_survey() -> Dict[str, Any]:
        """Create demographic survey for participants."""
        return {
            'age_range': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
            'education': [
                'High school',
                'Some college',
                'Bachelor\'s degree',
                'Master\'s degree',
                'Doctoral degree'
            ],
            'technical_expertise': [
                'Beginner',
                'Intermediate',
                'Advanced',
                'Expert'
            ],
            'email_frequency': [
                'Rarely',
                'Few times a week',
                'Daily',
                'Multiple times per day'
            ],
            'phishing_experience': [
                'Never heard of it',
                'Heard of it, not sure what it is',
                'Basic understanding',
                'Advanced understanding'
            ],
            'security_training': ['Yes', 'No'],
            'previous_phishing_victim': ['Yes', 'No', 'Unsure']
        }

    @staticmethod
    def create_post_study_survey() -> List[str]:
        """Create post-study questionnaire."""
        return [
            "How satisfied were you with the explanations overall? (1-5)",
            "Would you use this system in your daily email checking? (Yes/No)",
            "What did you like most about the explanations?",
            "What did you like least?",
            "How could the explanations be improved?",
            "Any other comments?"
        ]
