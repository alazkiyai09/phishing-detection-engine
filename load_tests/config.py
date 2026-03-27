"""
Load test configuration for Locust.
"""

# Target configuration
TARGET_HOST = "http://localhost:8000"

# User configuration
USERS_MIN = 10  # Minimum concurrent users
USERS_MAX = 100  # Maximum concurrent users
SPAWN_RATE = 10  # Users spawned per second
RUN_TIME = "5m"  # Test duration

# Performance targets
TARGET_RPS = 100  # Requests per second
P95_LATENCY_MS = 1000  # P95 latency target (1s)

# Scenarios
SCENARIOS = {
    "email_analysis": {
        "weight": 3,  # 60% of traffic
        "description": "Full email analysis"
    },
    "url_analysis": {
        "weight": 1,  # 20% of traffic
        "description": "Quick URL check"
    },
    "health_check": {
        "weight": 1,  # 20% of traffic
        "description": "Health/status endpoints"
    }
}
