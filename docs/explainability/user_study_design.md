# User Study Design

Complete protocol for evaluating explanation quality with human participants.

## Study Overview

**Title**: Evaluation of Human-Aligned Phishing Explanations

**Objective**: Evaluate whether explanations following human cognitive processing patterns are more effective than traditional technical explanations.

**Duration**: 45 minutes per participant

**Compensation**: $20 gift card

**Target Sample Size**: 30 participants

## Research Questions

### Primary Questions
1. **RQ1**: Do human-aligned explanations (cognitive order) improve understandability compared to traditional explanations?
2. **RQ2**: Do counterfactual explanations improve actionability?
3. **RQ3**: Does non-technical language increase user trust?
4. **RQ4**: Do explanations help users make better phishing detection decisions?

### Hypotheses
- **H1**: Explanations following cognitive order (sender→subject→body→URL) will be more understandable than random order
- **H2**: Counterfactual explanations will improve actionability compared to feature-based alone
- **H3**: Non-technical language will improve understandability for non-experts
- **H4**: Explanations will increase trust in phishing detection system

## Participants

### Inclusion Criteria
- 18 years or older
- Use email regularly (at least few times per week)
- No prior training in cybersecurity
- Fluent in English

### Exclusion Criteria
- Professional security analysts
- Previous participants in similar studies
- Vision impairments that affect screen reading (unless accommodated)

### Recruitment
- Posting on university campus
- Social media advertisements
- Community bulletin boards
- Participant pool (if available)

## Study Design

### Within-Subjects Design
Each participant evaluates:
- **10 emails total**:
  - 4 phishing emails
  - 3 safe emails
  - 3 suspicious emails

### Conditions
Participants see one of three explanation types (random assignment):
1. **Human-Aligned**: Cognitive order, non-technical language
2. **Technical**: Feature-based, technical metrics
3. **Mixed**: Combination of both

### Counterbalancing
- Email order randomized
- Explanation type between-subjects to avoid learning effects

## Procedure

### Phase 1: Introduction (5 minutes)

**Informed Consent**
- Explain study purpose
- Describe data collection
- Emphasize voluntary participation
- Right to withdraw

**Pre-Survey** (Demographics)
- Age range
- Education level
- Technical expertise
- Email frequency
- Phishing experience
- Security training
- Previous victim status

### Phase 2: Training (10 minutes)

**Example Explanations**
- Show 2 example emails with explanations
- Walk through interface
- Explain Likert scale (1-5)
- Answer questions

**Practice Task**
- 1 practice email (not counted)
- Ensure understanding

### Phase 3: Main Task (25 minutes)

**For each of 10 emails:**

1. **Present Email**
   - Show full email headers and body
   - 30 seconds to read

2. **Initial Decision** (without explanation)
   - "Do you think this is phishing or safe?"
   - Confidence in decision (1-5)
   - Time recorded

3. **Present Explanation**
   - Show explanation (condition-specific)
   - Can scroll and read
   - Time recorded

4. **Final Decision** (after explanation)
   - "Now do you think this is phishing or safe?"
   - Confidence in decision (1-5)
   - Time recorded

5. **Explanation Evaluation** (Likert scales)
   - Understandability: 4 questions
   - Helpfulness: 4 questions
   - Trust: 4 questions
   - Actionability: 3 questions
   - Total: 15 questions per email

6. **Optional Qualitative Feedback**
   - "What was confusing about this explanation?"
   - "What was helpful?"

**Total**: 10 emails × ~2.5 minutes each = 25 minutes

### Phase 4: Post-Study Survey (5 minutes)

**System-Level Questions**
- Overall satisfaction (1-5)
- Would you use this system? (Yes/No)
- What did you like most?
- What did you like least?
- How could it be improved?
- Any other comments?

**Debrief**
- Explain true purpose of study
- Show correct answers for all emails
- Provide resources about phishing
- Thank participant

## Measures

### Independent Variables
- **Explanation Type**: Human-aligned vs. Technical vs. Mixed
- **Email Type**: Phishing vs. Safe vs. Suspicious
- **Participant Expertise**: Beginner vs. Intermediate vs. Advanced

### Dependent Variables

#### Decision Quality
- **Accuracy**: Percentage of correct decisions
- **Decision Change**: Did explanation change their mind?
- **Confidence Calibration**: Does confidence match accuracy?

#### Timing
- **Time to Initial Decision**: Seconds
- **Time to Understand Explanation**: Seconds
- **Total Time**: Seconds

#### Subjective Ratings (1-5 Likert)
- **Understandability**:
  1. "I understood why the email was flagged"
  2. "The explanation was clear and easy to follow"
  3. "The technical language was understandable"
  4. "I could explain this to another person"

- **Helpfulness**:
  1. "The explanation helped me make a decision"
  2. "I learned something about phishing from this explanation"
  3. "The explanation addressed my concerns"
  4. "The information was relevant"

- **Trust**:
  1. "I trust the explanation provided"
  2. "The explanation seems accurate"
  3. "I believe the system is reliable"
  4. "I would use this system again"

- **Actionability**:
  1. "I know what action to take based on this explanation"
  2. "The explanation provides clear next steps"
  3. "I can apply this knowledge to future emails"

#### Qualitative Feedback
- Open-ended responses
- Coded for themes

## Materials

### Email Set
See `data/sample_emails.json` for complete set.

### Stimuli

#### Human-Aligned Explanation Example:
```
⚠️ This email is PHISHING (high confidence)

What We Found:
- Suspicious sender: Domain mimics Netflix
- Urgency in subject: "URGENT", "will be suspended"
- Suspicious link: Uses HTTP (not HTTPS)

What Should You Do:
- Don't click any links
- Report to IT security
- Contact Netflix directly
```

#### Technical Explanation Example:
```
Classification: PHISHING (AUPRC: 0.92)

Feature Importance:
- sender_lookalike_domain: 0.45
- subject_urgency_tfidf: 0.32
- url_http_flag: 0.28

Model Attention Weights:
- "suspended": 0.18
- "verify": 0.15
- "click": 0.12

Confidence Threshold: > 0.80
```

## Data Analysis Plan

### Pre-Analysis
1. **Data Cleaning**: Remove incomplete responses
2. **Normality Check**: Shapiro-Wilk test
3. **Outlier Detection**: Box plots, IQR method

### Primary Analyses

#### RQ1: Understandability
- **Test**: One-way ANOVA (explanation type → understandability)
- **Effect Size**: η²
- **Post-hoc**: Tukey HSD

#### RQ2: Actionability
- **Test**: Independent t-test (counterfactual vs. no counterfactual)
- **Effect Size**: Cohen's d

#### RQ3: Non-Technical Language
- **Test**: Two-way ANOVA (language × expertise)
- **Interaction**: Language simplification helps non-experts more

#### RQ4: Decision Accuracy
- **Test**: Repeated measures ANOVA (initial vs. final)
- **Improvement**: Gain score analysis

### Secondary Analyses
- **Confidence Calibration**: Brier score
- **Time Analysis**: Linear mixed models
- **Qualitative Coding**: Thematic analysis

### Statistical Power
- **Sample Size**: 30 participants × 10 emails = 300 observations
- **Power**: 0.80 to detect medium effect (d = 0.5)
- **Significance Level**: α = 0.05

## Ethical Considerations

### IRB Approval
- Submit to institutional IRB
- Category: Minimal risk
- Vulnerable populations: None targeted

### Informed Consent
- Written consent form
- Explicit permission for audio/video recording (if used)
- Data storage and retention policies

### Privacy Protection
- Anonymize participant IDs
- Store data on encrypted server
- No IP addresses collected
- Data retention: 3 years, then destroyed

### Debriefing
- Explain true purpose
- Provide educational materials
- Offer compensation regardless of completion
- Contact information for questions

### Risk Mitigation
- No deception used
- No sensitive PII collected
- Participants can skip any question
- Can withdraw at any time

## Timeline

| Week | Activity |
|------|----------|
| 1-2 | IRB approval |
| 3-4 | Material preparation |
| 5-8 | Participant recruitment |
| 9-12 | Data collection |
| 13-14 | Data analysis |
| 15 | Report writing |

## Budget

| Item | Cost |
|------|------|
| Participant compensation (30 × $20) | $600 |
| Online platform (e.g., Prolific) | $50 |
| Hosting/software | $0 (university) |
| **Total** | **$650** |

## Expected Outcomes

### Contributions
1. **Empirical evidence** for cognitive order in explanations
2. **Design guidelines** for phishing explanations
3. **Open-source implementation** for research community

### Publications
1. **CHI 2026**: Full paper on user study results
2. **USENIX Security**: Short paper on security implications
3. **arXiv**: Preprint with open-source code

### Impact
- Improved phishing detection for end-users
- Reduced victimization through better explanations
- Framework for other security explanations

---

## Appendix: Survey Questions

### Pre-Study Demographic Survey
```json
{
  "age_range": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
  "education": ["High school", "Some college", "Bachelor's", "Master's", "Doctoral"],
  "technical_expertise": ["Beginner", "Intermediate", "Advanced", "Expert"],
  "email_frequency": ["Rarely", "Few times/week", "Daily", "Multiple times/day"],
  "phishing_experience": ["Never heard", "Basic", "Advanced"],
  "security_training": ["Yes", "No"],
  "previous_victim": ["Yes", "No", "Unsure"]
}
```

### Post-Study Questions
1. How satisfied were you with the explanations overall? (1-5)
2. Would you use this system in your daily email checking? (Yes/No)
3. What did you like most about the explanations?
4. What did you like least?
5. How could the explanations be improved?
6. Any other comments?

---

For more information, contact: [your-email@example.com]
